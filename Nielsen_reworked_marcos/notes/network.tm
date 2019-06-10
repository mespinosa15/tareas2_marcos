<TeXmacs|1.99.8>

<style|generic>

<\body>
  We consider a neuron network of <math|L> layers. The activation states of
  neurons in the <math|l+1> layer is determined by the activation states of
  neurons in the <math|l> layer as follows:

  <\equation*>
    a<rsup|l+1><rsub|i>=\<sigma\><around*|(|<big|sum><rsub|j=1><rsup|n<rsup|l>>w<rsub|i
    j><rsup|l>a<rsub|j><rsup|l>+b<rsub|i><rsup|l>|)>=\<sigma\><around*|(|z<rsup|l><rsub|i>|)>,<space|2em>l=1,\<ldots\>,L-1
  </equation*>

  where <math|n<rsup|l>> is the number of neurons in layer <math|l>. The
  network input is given by the activation states of the neurons in the first
  layer, <math|a<rsub|1><rsup|1>,\<ldots\>,a<rsub|n<rsub|1>><rsup|1>>. The
  network parameters are the weights <math|w<rsub|i j><rsup|l>> and biases
  <math|b<rsub|i><rsup|l>>. It is convenient to define the \Pweighted
  inputs\Q <math|z<rsub|i><rsup|l>=<big|sum><rsub|j>w<rsub|i
  j><rsup|l>a<rsub|j><rsup|l>+b<rsub|i><rsup|l>>.

  The cost is defined as a function of the activation states of the last
  layer,

  <\equation*>
    C=C<around*|(|a<rsup|L><rsub|1>,\<ldots\>,a<rsub|n<rsub|L>><rsup|L>|)>
  </equation*>

  Since in turn <math|a<rsup|L><rsub|1>,\<ldots\>,a<rsub|n<rsub|L>><rsup|L>>
  are functions of the activation states of the previous layer, and so on
  until the first layer, we have the following recursion:

  <\equation*>
    <frac|\<partial\>C|\<partial\>a<rsub|j><rsup|l>>=<big|sum><rsub|i><frac|\<partial\>C|\<partial\>a<rsub|i><rsup|l+1>>\<sigma\><rprime|'><around*|(|z<rsub|i><rsup|l>|)>w<rsub|i
    j><rsup|l>
  </equation*>

  Finally, since <math|a<rsub|1><rsup|l+1>,\<ldots\>,a<rsub|n<rsub|l+1>><rsup|l+1>>
  are functions of the network parameters <math|b<rsub|i><rsup|l>,w<rsub|i
  j><rsup|l>>, we obtain:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>C|\<partial\>b<rsup|l><rsub|i>>>|<cell|=>|<cell|<frac|\<partial\>C|\<partial\>a<rsup|l+1><rsub|i>>\<sigma\><rprime|'><around*|(|z<rsup|l><rsub|i>|)>>>|<row|<cell|<frac|\<partial\>C|\<partial\>w<rsup|l><rsub|i
    j>>>|<cell|=>|<cell|<frac|\<partial\>C|\<partial\>a<rsup|l+1><rsub|i>>\<sigma\><rprime|'><around*|(|z<rsup|l><rsub|i>|)>a<rsub|j><rsup|l>>>>>
  </eqnarray*>

  Combining these equations with the recursion above, we obtain first that
  <math|<frac|\<partial\>C|\<partial\>a<rsub|j><rsup|l>>=<big|sum><rsub|i><frac|\<partial\>C|\<partial\>b<rsup|l><rsub|i>>w<rsub|i
  j><rsup|l>>, and therefore

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>C|\<partial\>b<rsup|l><rsub|i>>>|<cell|=>|<cell|<big|sum><rsub|k><frac|\<partial\>C|\<partial\>b<rsup|l+1><rsub|k>>w<rsub|k
    i><rsup|l+1>\<sigma\><rprime|'><around*|(|z<rsup|l><rsub|i>|)>>>|<row|<cell|<frac|\<partial\>C|\<partial\>w<rsup|l><rsub|i
    j>>>|<cell|=>|<cell|<big|sum><rsub|k><frac|\<partial\>C|\<partial\>b<rsup|l+1><rsub|k>>w<rsub|k
    i><rsup|l+1>\<sigma\><rprime|'><around*|(|z<rsup|l><rsub|i>|)>a<rsub|j><rsup|l>=<frac|\<partial\>C|\<partial\>b<rsup|l><rsub|i>>a<rsub|j><rsup|l>>>>>
  </eqnarray*>

  The backpropagation algorithm consists of iterating this recursion to
  compute the gradient with respect to all the parameters. This iteration
  begins from the gradient with respect to the last layer parameters:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>C|\<partial\>b<rsup|L-1><rsub|i>>>|<cell|=>|<cell|<frac|\<partial\>C|\<partial\>a<rsup|L><rsub|i>>\<sigma\><rprime|'><around*|(|z<rsup|L-1><rsub|i>|)>>>|<row|<cell|<frac|\<partial\>C|\<partial\>w<rsup|L-1><rsub|i
    j>>>|<cell|=>|<cell|<frac|\<partial\>C|\<partial\>a<rsup|L><rsub|i>>\<sigma\><rprime|'><around*|(|z<rsup|L-1><rsub|i>|)>a<rsub|j><rsup|L-1>=<frac|\<partial\>C|\<partial\>b<rsup|L-1><rsub|i>>a<rsub|j><rsup|L-1>>>>>
  </eqnarray*>

  which can be computed directly from the form of the cost function.

  As in Nielsen's book, it is convenient to define

  <\equation*>
    \<delta\><rsub|i><rsup|l>=<frac|\<partial\>C|\<partial\>z<rsub|i><rsup|l>>=<frac|\<partial\>C|\<partial\>a<rsub|i><rsup|l+1>>\<sigma\><rprime|'><around*|(|z<rsub|i><rsup|l>|)>,<space|2em>l=1,\<ldots\>,L-1
  </equation*>

  Then

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>C|\<partial\>b<rsup|l><rsub|i>>>|<cell|=>|<cell|\<delta\><rsub|i><rsup|l>>>|<row|<cell|<frac|\<partial\>C|\<partial\>w<rsup|l><rsub|i
    j>>>|<cell|=>|<cell|\<delta\><rsub|i><rsup|l>a<rsub|j><rsup|l>>>>>
  </eqnarray*>

  and

  <\equation*>
    \<delta\><rsub|i><rsup|l-1>=<big|sum><rsub|k>\<delta\><rsub|k><rsup|l>w<rsub|k
    i><rsup|l>
  </equation*>
</body>

<initial|<\collection>
</collection>>