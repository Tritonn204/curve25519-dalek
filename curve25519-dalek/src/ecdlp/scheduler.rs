use std::thread::{Result, Scope, ScopedJoinHandle};

/// A trait representing a handle to a spawned task.
pub trait TaskHandle<'scope, R> {
    /// Wait for the task to complete and retrieve its result.
    fn join(self) -> Result<R>;
}

/// A trait representing a scope in which tasks can be spawned.
pub trait SchedulerScope<'scope, 'env: 'scope> {
    /// Spawn a new task within this scope.
    fn spawn<F, T>(&'scope self, f: F) -> impl TaskHandle<'scope, T>
    where
        T: Send + 'env,
        F: FnOnce() -> T + Send + 'env;
}

/// A trait representing a scheduler that can create scopes for spawning tasks.
pub trait Scheduler {
    /// The scope type associated with this scheduler.
    type Scope<'scope, 'env: 'scope>: SchedulerScope<'scope, 'env>;

    /// Create a new scope for spawning tasks.
    fn scope<'env, F, T>(scope: F) -> T
    where
        T: Send,
        F: for<'scope> FnOnce(&'scope Self::Scope<'scope, 'env>) -> T;
}

/// A standard implementation of the `Scheduler` trait using Rust's standard library threads.
pub struct DefaultScheduler;

impl Scheduler for DefaultScheduler {
    type Scope<'scope, 'env: 'scope> = Scope<'scope, 'env>;

    fn scope<'env, F, T>(scope: F) -> T
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
    {
        std::thread::scope(|s| scope(s))
    }
}

impl<'scope, R> TaskHandle<'scope, R> for ScopedJoinHandle<'scope, R> {
    fn join(self) -> Result<R> {
        self.join()
    }
}

impl<'scope, 'env: 'scope> SchedulerScope<'scope, 'env> for Scope<'scope, 'env> {
        fn spawn<F, T>(&'scope self, f: F) -> impl TaskHandle<'scope, T>
        where
        T: Send + 'env,
        F: FnOnce() -> T + Send + 'env,
    {
        self.spawn(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_scheduler() {
        fn test<S: Scheduler>() -> usize {
            S::scope(|s| {
                let handle = s.spawn(|| 42);
                handle.join().expect("Task panicked")
            })
        }

        assert_eq!(test::<DefaultScheduler>(), 42);
    }
}