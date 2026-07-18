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

#[derive(Debug)]
struct EventCount {
    waiters: AtomicUsize,
    epoch: Mutex<Option<usize>>,
    cond: Condvar,
}

struct Waiter<'a> {
    ec: &'a EventCount,
    epoch: usize,
}

impl<'a> Waiter<'a> {
    fn new(ec: &'a EventCount) -> Self {
        ec.waiters.fetch_add(1, Ordering::Relaxed);
        Self { ec, epoch: 0 }
    }
}

impl Drop for Waiter<'_> {
    fn drop(&mut self) {
        self.ec.waiters.fetch_sub(1, Ordering::Relaxed);
    }
}

impl EventCount {
    pub fn new() -> Self {
        Self {
            waiters: AtomicUsize::new(0),
            epoch: Mutex::new(Some(0)),
            cond: Condvar::new(),
        }
    }

    /// Announce intent to sleep, *before* the final condition check.
    ///
    /// Returns `None` if shutdown is signaled.
    pub fn prepare_wait<'a>(&'a self) -> Option<Waiter<'a>> {
        let mut w = Waiter::new(self);
        atomic::fence(Ordering::SeqCst);
        if let Some(epoch) = *self.epoch.lock().unwrap() {
            w.epoch = epoch;
            return Some(w);
        }
        None
    }

    /// Re-check found nothing: block until the epoch moves past `w`.
    ///
    /// Returns `false` if shutdown is signaled.
    pub fn wait<'a>(&'a self, w: Waiter<'a>) -> bool {
        let mut guard = self.epoch.lock().unwrap();
        while let Some(epoch) = *guard {
            if epoch == w.epoch {
                guard = self.cond.wait(guard).unwrap();
                continue;
            }
            return true;
        }
        false
    }

    /// Producer: call *after* publishing work to the deque.
    pub fn notify_one(&self) {
        atomic::fence(Ordering::SeqCst);
        if self.waiters.load(Ordering::Relaxed) != 0 {
            let mut guard = self.epoch.lock().unwrap();
            if let Some(epoch) = *guard {
                *guard = Some(epoch.wrapping_add(1));
                self.cond.notify_one();
            }
        }
    }

    /// Producer: signal shutdown.
    pub fn notify_shutdown(&self) {
        let mut guard = self.epoch.lock().unwrap();
        *guard = None;
        self.cond.notify_all();
    }
}

/// Handed to the seeding closure and the task handler so they can enqueue work.
/// Holds raw pointers but the branded lifetime `'scope` makes sure they cannot
/// outlive the underlying local queue or the shared control block.
#[derive(Debug)]
pub struct Scope<'s> {
    local: NonNull<Worker<usize>>,
    shared: NonNull<Shared>,
    first: Cell<bool>,
    _brand: PhantomData<fn(&'s ()) -> &'s ()>,
}

impl<'s> Scope<'s> {
    pub fn spawn(&self, task: usize) {
        let shared = unsafe { self.shared.as_ref() };
        let local = unsafe { self.local.as_ref() };
        shared.pending.fetch_add(1, Ordering::Relaxed);
        local.push(task);
        // First spawn stays for this thread to run; only the surplus recruits.
        if !self.first.replace(false) {
            shared.ec.notify_one();
        }
    }
}

/// Shared control block of the thread pool.
struct Shared {
    stealers: Vec<Stealer<usize>>,
    task_data: AtomicPtr<()>,
    task_fn: AtomicPtr<()>,
    pending: AtomicUsize,
    ec: EventCount,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

type TaskFn = unsafe fn(data: *const (), task: usize, scope: &Scope<'_>);

/// A work-stealing thread pool, with tasks indexed by `usize`.
pub struct Pool {
    local: Worker<usize>,
    shared: Arc<Shared>,
    other_threads: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Creates a new thread pool with the given number of worker threads.
    pub fn new(num_other_threads: usize) -> Self {
        // Build every task queue up front.
        let local = Worker::new_lifo();
        let other_locals = (0..num_other_threads)
            .map(|_| Worker::new_lifo())
            .collect::<Vec<_>>();
        let stealers = std::iter::once(&local)
            .chain(other_locals.iter())
            .map(Worker::stealer)
            .collect::<Vec<_>>();

        // Create the shared control block.
        let shared = Arc::new(Shared {
            stealers,
            task_data: AtomicPtr::new(std::ptr::null_mut()),
            task_fn: AtomicPtr::new(std::ptr::null_mut()),
            pending: AtomicUsize::new(0),
            ec: EventCount::new(),
            panic: Mutex::new(None),
        });

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
    pub fn run<F>(&mut self, seed: impl IntoIterator<Item = usize>, handler: F)
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
            first: Cell::new(true),
            _brand: PhantomData,
        };
        for task in seed {
            scope.spawn(task);
        }

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
        self.shared.ec.notify_shutdown();

        // Wait for every worker to exit.
        for t in self.other_threads.drain(..) {
            t.join().unwrap();
        }
    }
}

fn worker_fn(local: &Worker<usize>, shared: &Shared) {
    let backoff = Backoff::new();

    // Check for available tasks until backoff gives up.
    loop {
        match find_task(local, shared) {
            FindResult::Task(task) => {
                run_task(local, shared, task);
                backoff.reset();
            }
            _ if !backoff.is_completed() => backoff.snooze(),

            // Signalling sleepy state and re-check.
            _ => match shared.ec.prepare_wait() {
                Some(w) => match find_task(local, shared) {
                    FindResult::Task(task) => {
                        drop(w);
                        run_task(local, shared, task);
                        backoff.reset();
                    }
                    _ => match shared.ec.wait(w) {
                        true => backoff.reset(),
                        false => return,
                    },
                },
                None => return,
            },
        }
    }
}

enum FindResult {
    Task(usize),
    Pending,
    Done,
}

fn find_task(local: &Worker<usize>, shared: &Shared) -> FindResult {
    if let Some(task) = local.pop() {
        FindResult::Task(task)
    } else if shared.pending.load(Ordering::Acquire) == 0 {
        FindResult::Done
    } else if let Some(task) = shared.stealers.iter().find_map(|s| s.steal().success()) {
        FindResult::Task(task)
    } else {
        FindResult::Pending
    }
}

fn run_task(local: &Worker<usize>, shared: &Shared, task: usize) {
    // Run the task, then decrement the pending count.
    let scope = Scope {
        local: NonNull::from(local),
        shared: NonNull::from(shared),
        first: Cell::new(true),
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
    shared.pending.fetch_sub(1, Ordering::Release);
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
