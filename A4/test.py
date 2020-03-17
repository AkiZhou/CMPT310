def remainder_is_even(num, div):
    x = (num % div) % 2
    if x != 0:
        result = 'false'
    else:
        result = 'true'

    