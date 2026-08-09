//! The thread pool.

use std::any::Any;
use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam::deque::{Stealer, Worker};
use crossbeam::utils::Backoff;

/// Recruit one worker thread for every `LIGHT_PER_NOTIFY` light tasks, to
/// reduce the number of wakeups and scheduling overhead (single-threaded
/// node scheduling overhead is ~50ns, while waking up a new thread can take
/// orders of magnitudes longer).
const LIGHTS_PER_NOTIFY: usize = 1024;

/// Handed to the seeding closure and the task handler so they can enqueue work.
/// Holds raw pointers but the branded lifetime `'scope` makes sure they cannot
/// outlive the underlying local queue or the shared control block.
#[derive(Debug)]
pub struct Scope<'s> {
    local: NonNull<Local>,
    shared: NonNull<Shared>,
    _brand: PhantomData<fn(&'s ()) -> &'s ()>,
}

impl<'s> Scope<'s> {
    pub fn spawn(&self, task: usize, is_heavy: bool) {
        let shared = unsafe { self.shared.as_ref() };
        let local = unsafe { self.local.as_ref() };
        shared.pending.fetch_add(1, Ordering::Relaxed);
        local.queue.push(task);
        if is_heavy {
            shared.cond.notify_one();
        } else {
            local.lights.update(|n| n.wrapping_add(1));
            if local.lights.get().is_multiple_of(LIGHTS_PER_NOTIFY) {
                shared.cond.notify_one();
            }
        }
    }
}

/// Local state of a worker thread.
struct Local {
    queue: Worker<usize>,
    lights: Cell<usize>,
}

impl Local {
    fn new() -> Self {
        Self {
            queue: Worker::new_lifo(),
            lights: Cell::new(0),
        }
    }
}

/// Shared control block of the thread pool.
struct Shared {
    stealers: Vec<Stealer<usize>>,
    task_data: AtomicPtr<()>,
    task_fn: AtomicPtr<()>,
    pending: AtomicUsize,
    cond: Condvar,
    shutdown: Mutex<bool>,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Shared {
    fn new(stealers: Vec<Stealer<usize>>) -> Self {
        Self {
            stealers,
            task_data: AtomicPtr::new(std::ptr::null_mut()),
            task_fn: AtomicPtr::new(std::ptr::null_mut()),
            pending: AtomicUsize::new(0),
            cond: Condvar::new(),
            shutdown: Mutex::new(false),
            panic: Mutex::new(None),
        }
    }
}

type TaskFn = unsafe fn(data: *const (), task: usize, scope: &Scope<'_>);

/// A work-stealing thread pool, with tasks indexed by `usize`.
pub struct Pool {
    local: Local,
    shared: Arc<Shared>,
    other_threads: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Creates a new thread pool with the given number of worker threads.
    pub fn new(num_other_threads: usize) -> Self {
        // Build every task queue up front.
        let local = Local::new();
        let other_locals = (0..num_other_threads)
            .map(|_| Local::new())
            .collect::<Vec<_>>();
        let stealers = std::iter::once(&local)
            .chain(other_locals.iter())
            .map(|local| local.queue.stealer())
            .collect::<Vec<_>>();

        // Create the shared control block.
        let shared = Arc::new(Shared::new(stealers));

        // Spawn the worker threads.
        let other_threads = other_locals
            .into_iter()
            .map(|local| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || worker_fn(&local, &shared))
            })
            .collect();

        Pool {
            local,
            shared,
            other_threads,
        }
    }

    /// Returns the number of worker threads in the pool.
    pub fn num_other_threads(&self) -> usize {
        self.other_threads.len()
    }

    /// Run a scope on the pool. `seed` enqueues the initial tasks; `handler`
    /// processes one task and may enqueue more via `Scope::spawn`. Blocks
    /// until every task (including transitively spawned ones) completes.
    ///
    /// Takes `&mut self`, so the borrow checker enforces one scope at a time:
    /// concurrent runs and nested runs (a handler re-entering `run`) are both
    /// compile errors -- which is why the single global batch state below is
    /// sound. The pool is freely reusable across *sequential* runs and graphs.
    /// Running graphs in parallel on a shared pool would instead need per-scope
    /// state (a completion latch + handler per scope, scope-tagged tasks) and a
    /// `&self` signature.
    pub fn run<F>(&mut self, seed: impl FnOnce(&Scope<'_>), handler: F)
    where
        F: for<'s> Fn(usize, &Scope<'s>) + Sync,
    {
        // Publish the (lifetime-erased) task procedure.
        let task_data = task_data_for(&handler) as *mut ();
        let task_fn = task_fn_for::<F> as TaskFn as *mut ();
        self.shared.task_data.store(task_data, Ordering::Relaxed);
        self.shared.task_fn.store(task_fn, Ordering::Relaxed);

        // Seed initial tasks into the global queue.
        let scope = Scope {
            local: NonNull::from(&self.local),
            shared: NonNull::from(&*self.shared),
            _brand: PhantomData,
        };
        seed(&scope);

        // Check for available tasks until none left. Main thread never sleeps.
        let backoff = Backoff::new();
        loop {
            match find_task(&self.local, &self.shared) {
                FindResult::Task(task) => {
                    run_task(&self.local, &self.shared, task);
                    backoff.reset();
                }
                FindResult::Done => break,
                _ => backoff.snooze(),
            }
        }

        // A handler panicked (it was caught, so no worker died and `pending`
        // still reached 0); re-raise it now on the caller's thread. Drop the
        // guard *before* re-raising -- unwinding while holding it would poison
        // the mutex and make the next `run` panic on lock.
        let caught = self.shared.panic.lock().unwrap().take();
        if let Some(payload) = caught {
            std::panic::resume_unwind(payload);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Signal shutdown via mutex and wake all the workers so they can exit.
        *self.shared.shutdown.lock().unwrap() = true;
        self.shared.cond.notify_all();

        // Wait for every worker to exit.
        for t in self.other_threads.drain(..) {
            t.join().unwrap();
        }
    }
}

fn worker_fn(local: &Local, shared: &Shared) {
    let backoff = Backoff::new();

    // Check for available tasks until backoff gives up.
    loop {
        match find_task(local, shared) {
            FindResult::Task(task) => {
                run_task(local, shared, task);
                backoff.reset();
            }
            _ if !backoff.is_completed() => backoff.snooze(),
            _ => {
                let guard = shared.shutdown.lock().unwrap();
                if *guard {
                    return;
                }
                drop(shared.cond.wait(guard).unwrap());
                backoff.reset();
            }
        }
    }
}

enum FindResult {
    Task(usize),
    Pending,
    Done,
}

fn find_task(local: &Local, shared: &Shared) -> FindResult {
    if let Some(task) = local.queue.pop() {
        FindResult::Task(task)
    } else if shared.pending.load(Ordering::Relaxed) == 0 {
        atomic::fence(Ordering::Acquire);
        FindResult::Done
    } else if let Some(task) = shared
        .stealers
        .iter()
        .find_map(|s| s.steal_batch_and_pop(&local.queue).success())
    {
        FindResult::Task(task)
    } else {
        FindResult::Pending
    }
}

fn run_task(local: &Local, shared: &Shared, task: usize) {
    // Run the task, then decrement the pending count.
    let scope = Scope {
        local: NonNull::from(local),
        shared: NonNull::from(shared),
        _brand: PhantomData,
    };
    let task_data = shared.task_data.load(Ordering::Relaxed);
    let task_fn = shared.task_fn.load(Ordering::Relaxed);

    // Catch a panicking handler so this worker survives and `pending` is always
    // decremented (a skipped decrement would hang `run` forever). Stash the
    // first payload *before* the decrement, so a thread that observes
    // `pending == 0` is guaranteed to also see the payload `run` re-raises.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        std::mem::transmute::<*mut (), TaskFn>(task_fn)(task_data, task, &scope)
    }));
    if let Err(payload) = result {
        let mut slot = shared.panic.lock().unwrap();
        if slot.is_none() {
            *slot = Some(payload);
        }
    }
    atomic::fence(Ordering::Release);
    shared.pending.fetch_sub(1, Ordering::Relaxed);
}

fn task_data_for<F>(handler: &F) -> *const F
where
    F: for<'s> Fn(usize, &Scope<'s>) + Sync,
{
    handler as *const F
}

unsafe fn task_fn_for<F>(data: *const (), task: usize, scope: &Scope<'_>)
where
    F: for<'s> Fn(usize, &Scope<'s>) + Sync,
{
    let closure = unsafe { &*(data as *const F) };
    closure(task, scope);
}
