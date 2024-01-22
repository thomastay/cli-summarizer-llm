import time


def timing(func, verbose=False):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if verbose:
            print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper
