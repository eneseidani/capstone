# S05 random subset — Windows log lock correction

This patch **does not change model architectures, subset sampling, or chronological validation**. It changes logging lifetime only.

## Fixed portion 1: log setup and cleanup

```python
def close_logging_handlers():
    """Release log files deterministically, including after failed test runs.

    Windows does not allow TemporaryDirectory to remove an open FileHandler.
    Remove each handler from this script's logger, flush, then close it.
    """
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


def setup_logging(out):
    out.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    close_logging_handlers()
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    fh = logging.FileHandler(out / 'overnight.log', encoding='utf-8')
    fh.setFormatter(formatter)
    LOG.addHandler(stream)
    LOG.addHandler(fh)
```

## Fixed portion 2: guaranteed cleanup on run exit

The former `run(args)` implementation is now `_run_internal(args)`. This wrapper is added to ensure log handles close after successful runs and failures:

```python
def run(args):
    """Run experiments and always close log files on return or exception."""
    try:
        return _run_internal(args)
    except Exception:
        LOG.exception('Overnight experiment stopped. Fix the error and rerun the exact same command to resume.')
        raise
    finally:
        close_logging_handlers()


if __name__=='__main__':
    run(parse_args())
```

The complete updated file is `S05_random_subset_overnight.py` alongside this note. The updated `test_random_subset_overnight.py` adds a regression test for cleanup after a missing dependency, in addition to the original end-to-end resume test.
