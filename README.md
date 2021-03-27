# TuckER

## Using the TuckER class

This class fully supports batching.
```
tucker = TuckER(num_entities, num_relations, initial_tensor, gradient_mask=None)
```

```initial_tensor``` is what the ```core_tensor``` is initialized with.
The ```gradient_mask``` entry must have the same shape as the ```initial_tensor```.
For most purposes, the user may leave the ```gradient_mask``` attribute as ```None```.
In such cases, it is initialized to a tensor of ones with the same shape as ```initial_tensor```.

Example:

```
tucker = TuckER(9, 9, np.random.normal(size=[3, 11, 3]).astype(np.float32), np.ones([3, 11, 3]).astype(np.float32))
output = tucker([0, 1], [5, 2])
print(output)
```

produces

```
tensor([[1.3786e-02, 9.9045e-01, 2.6718e-02, 1.1845e-01, 7.9078e-01, 9.9812e-01,
         1.0000e+00, 4.2361e-03, 9.6301e-01],
        [9.9839e-01, 9.8582e-01, 1.3405e-05, 9.2480e-01, 9.9501e-01, 9.8919e-01,
         9.9992e-01, 4.7132e-01, 1.4876e-03]], grad_fn=<SigmoidBackward>)
```

## Using TuckER to implement other knowledge-graph completion algorithms

Here is where ```gradient_mask``` comes in.
Whichever values in the ```core_tensor``` should not be modified should have an entry of ```0```
in the ```gradient_mask``` and other values should be ```1```.