---
layout: home
---

Represent code expressions as data structures, then transform them.

## Install

```bash
pip install vexpr
```

## Get started


<style>
.hidden {
    display: none;
}

.tab-contents {
    border: 1px solid black;
    padding-top: 10px;
}

.tab-contents h4 {
    margin-top: 10px;
    margin-left: 10px;
    font-weight: bold;
}

.tab-contents :last-child {
    margin-bottom: 0;
}

.tabs {
    overflow: hidden;
}

.tab-label {
    background-color:darkgray;
    float: left;
    text-align: center;
    padding: 5px 30px;
    border: 1px solid black;

    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
}

/* on tab-label hover when not selected */
.tab-label:hover:not(.selected) {
    background-color: #ddd;
    cursor: pointer;
}

.tab-label.selected {
    background-color: #f1f1f1;
    cursor: default;
}
</style>

<script>

function showNumpyCode() {
    document.getElementById("numpy-code").classList.remove("hidden");
    document.getElementById("pytorch-code").classList.add("hidden");
    document.getElementById("jax-code").classList.add("hidden");

    document.getElementById("numpy-tab").classList.add("selected");
    document.getElementById("torch-tab").classList.remove("selected");
    document.getElementById("jax-tab").classList.remove("selected");
}

function showTorchCode() {
    document.getElementById("numpy-code").classList.add("hidden");
    document.getElementById("pytorch-code").classList.remove("hidden");
    document.getElementById("jax-code").classList.add("hidden");

    document.getElementById("numpy-tab").classList.remove("selected");
    document.getElementById("torch-tab").classList.add("selected");
    document.getElementById("jax-tab").classList.remove("selected");
}

function showJaxCode() {
    document.getElementById("numpy-code").classList.add("hidden");
    document.getElementById("pytorch-code").classList.add("hidden");
    document.getElementById("jax-code").classList.remove("hidden");

    document.getElementById("numpy-tab").classList.remove("selected");
    document.getElementById("torch-tab").classList.remove("selected");
    document.getElementById("jax-tab").classList.add("selected");
}

</script>

Example: a custom distance metric between two lists of vectors, `x1` and `x2`.

<div class="tabs">
    <!-- three tabs, one for each of NumPy / PyTorch / JAX -->

    <div id="numpy-tab" class="tab-label selected" onclick="showNumpyCode()">NumPy</div>
    <div id="torch-tab" class="tab-label" onclick="showTorchCode()">PyTorch</div>
    <div id="jax-tab" class="tab-label" onclick="showJaxCode()">JAX</div>
</div>
<div class="tab-contents">
<div id="numpy-code">

<h4>1. Create a Vexpr</h4>

{% highlight python %}
import vexpr as vp
import vexpr.numpy as vnp
import vexpr.scipy.spatial.distance as vsd

w1 = vp.symbol("w1")
w2 = vp.symbol("w2")
x1 = vp.symbol("x1")
x2 = vp.symbol("x2")

expr = vnp.sum([w1 * vsd.cdist(x1[..., [0, 1, 2]], x2[..., [0, 1, 2]]),
                w2 * vsd.cdist(x1[..., [0, 3, 4]], x2[..., [0, 3, 4]])],
               axis=0)
print(expr)
# Output: a Vexpr data structure:
#
# numpy.sum(
#   [operator.mul(
#     symbol('w1'),
#     scipy.spatial.distance.cdist(
#       operator.getitem(
#         symbol('x1'),
#         (Ellipsis, [0, 1, 2]),
#       ),
#       operator.getitem(
#         symbol('x2'),
#         (Ellipsis, [0, 1, 2]),
#       ),
#     ),
#   ),
#    operator.mul(
#     symbol('w2'),
#     scipy.spatial.distance.cdist(
#       operator.getitem(
#         symbol('x1'),
#         (Ellipsis, [0, 3, 4]),
#       ),
#       operator.getitem(
#         symbol('x2'),
#         (Ellipsis, [0, 3, 4]),
#       ),
#     ),
#   )]
#   axis=0
# )

{% endhighlight %}

<h4>2. Transform into a faster Vexpr that would have been difficult to write directly</h4>

{% highlight python %}
import numpy as np

example_inputs = dict(
    x1=np.random.randn(10, 5),
    x2=np.random.randn(10, 5),
    w1=np.array(0.7),
    w2=np.array(0.3),
)

expr = vp.vectorize(expr, example_inputs)
print(expr)
# numpy.sum(
#   operator.mul(
#     numpy.reshape(
#       numpy.stack([symbol('w1'), symbol('w2')]),
#       (2, 1, 1),
#     ),
#     custom.scipy.cdist_multi(
#       operator.getitem(
#         symbol('x1'),
#         (Ellipsis, array([0, 1, 2, 0, 3, 4])),
#       ),
#       operator.getitem(
#         symbol('x2'),
#         (Ellipsis, array([0, 1, 2, 0, 3, 4])),
#       ),
#       lengths=array([3, 3])
#     ),
#   )
#   axis=0
# )
{% endhighlight %}

<h4>3. Evaluate the Vexpr, as you would if you were training w1 and w2</h4>

{% highlight python %}
inputs = dict(x1=np.random.randn(12, 5),
              x2=np.random.randn(4, 5),
              w1=np.array(0.6),
              w2=np.array(0.4),)
print(vp.eval(expr, inputs))
# [[1.55860886 1.81932763 1.36601246 2.74558064]
#  [1.07449014 2.41388948 2.05383731 3.47491204]
#  [3.44607574 4.11058513 1.73149737 3.99700678]
#  [1.42342409 1.89316449 2.36516876 2.61242728]
#  [2.10589466 2.16815159 1.05028078 3.2819643 ]
#  [2.6376981  1.86969234 4.09429083 3.39908103]
#  [2.46510162 2.13610497 2.91302844 3.65995608]
#  [1.65351302 1.66339115 2.56035358 1.93349338]
#  [1.15303396 2.07962417 2.23623819 2.63961701]
#  [2.90055677 1.57172764 3.10181813 2.25698896]
#  [1.83600204 2.63654294 1.22630251 3.47381211]
#  [2.61149285 2.77062418 0.78998639 3.10032325]]
{% endhighlight %}

<h4>4. Use partial evaluation to precompute intermediate state, as you would before inference</h4>

{% highlight python %}
parameters = dict(w1=0.6, w2=0.4)
expr = vp.partial_eval(expr, parameters)
print(expr)
# numpy.sum(
#   operator.mul(
#     array([[[0.6]],
#            [[0.4]]]),
#     custom.scipy.cdist_multi(
#       operator.getitem(
#         symbol('x1'),
#         (Ellipsis, array([0, 1, 2, 0, 3, 4])),
#       ),
#       operator.getitem(
#         symbol('x2'),
#         (Ellipsis, array([0, 1, 2, 0, 3, 4])),
#       ),
#       lengths=array([3, 3])
#     ),
#   )
#   axis=0
# )
{% endhighlight %}

</div>

<div id="pytorch-code" class="hidden">

<h4>1. Create a Vexpr</h4>

{% highlight python %}
import vexpr as vp
import vexpr.torch as vtorch

w1 = vp.symbol("w1")
w2 = vp.symbol("w2")
x1 = vp.symbol("x1")
x2 = vp.symbol("x2")

expr = vtorch.sum([w1 * vtorch.cdist(x1[..., [0, 1, 2]], x2[..., [0, 1, 2]]),
                   w2 * vtorch.cdist(x1[..., [0, 3, 4]], x2[..., [0, 3, 4]])],
                  dim=0)
print(expr)
# torch.sum(
#   [operator.mul(
#     symbol('w1'),
#     torch.cdist(
#       operator.getitem(
#         symbol('x1'),
#         (Ellipsis, [0, 1, 2]),
#       ),
#       operator.getitem(
#         symbol('x2'),
#         (Ellipsis, [0, 1, 2]),
#       ),
#     ),
#   ),
#    operator.mul(
#     symbol('w2'),
#     torch.cdist(
#       operator.getitem(
#         symbol('x1'),
#         (Ellipsis, [0, 3, 4]),
#       ),
#       operator.getitem(
#         symbol('x2'),
#         (Ellipsis, [0, 3, 4]),
#       ),
#     ),
#   )]
#   dim=0
# )
{% endhighlight %}


<h4>2. Transform into a faster Vexpr that would have been difficult to write directly</h4>

{% highlight python %}
import torch

example_inputs = dict(
    x1=torch.randn(10, 5),
    x2=torch.randn(10, 5),
    w1=torch.tensor(0.7),
    w2=torch.tensor(0.3),
)

expr = vp.vectorize(expr, example_inputs)
print(expr)
# torch.sum(
#   operator.mul(
#     torch.reshape(
#       torch.stack([symbol('w1'), symbol('w2')]),
#       (2, 1, 1),
#     ),
#     custom.torch.cdist_multi(
#       custom.torch.split_and_stack(
#         operator.getitem(
#           symbol('x1'),
#           (Ellipsis, tensor([0, 1, 2, 0, 3, 4])),
#         )
#         lengths=[3, 3], expanded_length=6, expanded_indices=tensor([0, 1, 2, 3, 4, 5]), max_length=3, dim=-1
#       ),
#       custom.torch.split_and_stack(
#         operator.getitem(
#           symbol('x2'),
#           (Ellipsis, tensor([0, 1, 2, 0, 3, 4])),
#         )
#         lengths=[3, 3], expanded_length=6, expanded_indices=tensor([0, 1, 2, 3, 4, 5]), max_length=3, dim=-1
#       ),
#       p=2
#     ),
#   )
#   dim=0
# )
{% endhighlight %}

<h4>3. Evaluate the Vexpr, as you would if you were training w1 and w2</h4>

{% highlight python %}
inputs = dict(x1=torch.randn(12, 5),
              x2=torch.randn(4, 5),
              w1=torch.tensor(0.6),
              w2=torch.tensor(0.4),)
print(vp.eval(expr, inputs))
# tensor([[3.1750, 2.2383, 2.6217, 1.0710],
#         [2.3972, 1.8493, 1.8987, 1.8038],
#         [2.7758, 0.9884, 1.8191, 2.8204],
#         [1.5958, 2.4894, 2.3942, 2.3034],
#         [2.2631, 0.7308, 1.2725, 1.6628],
#         [2.7736, 0.8804, 1.8810, 2.0894],
#         [2.7475, 1.7807, 1.7098, 1.9817],
#         [1.6824, 2.3360, 2.4505, 2.4344],
#         [1.4595, 1.9179, 1.7824, 1.4457],
#         [2.1513, 1.6023, 0.9952, 1.4258],
#         [2.4210, 3.1545, 2.1091, 2.6089],
#         [3.2171, 1.3637, 2.2806, 3.0934]])
{% endhighlight %}

<h4>4. Use partial evaluation to precompute intermediate state, as you would before inference</h4>

{% highlight python %}
parameters = dict(w1=0.6, w2=0.4)
expr = vp.partial_eval(expr, parameters)
print(expr)
# torch.sum(
#   operator.mul(
#     tensor([[[0.6000]],
#             [[0.4000]]]),
#     custom.torch.cdist_multi(
#       custom.torch.split_and_stack(
#         operator.getitem(
#           symbol('x1'),
#           (Ellipsis, tensor([0, 1, 2, 0, 3, 4])),
#         )
#         lengths=[3, 3], expanded_length=6, expanded_indices=tensor([0, 1, 2, 3, 4, 5]), max_length=3, dim=-1
#       ),
#       custom.torch.split_and_stack(
#         operator.getitem(
#           symbol('x2'),
#           (Ellipsis, tensor([0, 1, 2, 0, 3, 4])),
#         )
#         lengths=[3, 3], expanded_length=6, expanded_indices=tensor([0, 1, 2, 3, 4, 5]), max_length=3, dim=-1
#       ),
#       p=2
#     ),
#   )
#   dim=0
# )
{% endhighlight %}


</div>

<div id="jax-code" class="hidden">

<div style="margin: 10px 20px 20px 20px;">Coming soon.</div>

</div>
</div>
