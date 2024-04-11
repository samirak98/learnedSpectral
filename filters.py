def filter(Pi, Delta, sigma, method = 'mse', beta = 0.375):
    if method == 'mse' or method == 'prox':
        return (sigma*Pi)/(sigma**2*Pi + Delta)
    elif method == 'sc':
        return (sigma*Pi)/(sigma**2*Pi + Delta/(8*beta))
    elif method == 'post':
        return (sigma*Pi)/(sigma**2*Pi + sigma**2*Pi*Delta)
    elif method == 'adv':
        return sigma/(sigma**2+ 3*Delta/(8*beta*(3*sigma**2*Pi+Delta)))
    elif method == 'nofilter':
        return 1/sigma
    else:
        raise NotImplementedError('The chosen filter method is not implemented. Please choose one of the following methods: \'mse\', \'post\', \'prox\', \'adv\', \'sc\', \'nofilter\'')