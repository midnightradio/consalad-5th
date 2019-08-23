import functools

def memoize(func):
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func

@memoize
def fib_rec(n):
    if n<=1:
        return n
    return fib_rec(n-1) + fib_rec(n-2)

def fib_mem(n, cache):
    if n == 0 or n == 1:
        cache[n] = n

    if n not in cache:
        cache[n] = fib_mem(n-1, cache) + fib_mem(n-2, cache)

    return cache[n]

def fib_tab(n):
    f = [0] * (n+1)
    f[1] = 1
    for i in range(2, n+1):
        f[i] = f[i-1] + f[i-2]
    return f[n]
