**Holla, Lets get more neighbours. Lets Scale the data and get more neighbours(pixels) with in a small range of distance.**

   ![download](https://user-images.githubusercontent.com/60026221/215568087-a5a603c3-d4e9-4641-ae21-095f08020d35.jpeg)

    - Generally, this is how normalization/standardization really works.

So, as you may got atleast a small idea what & why scaling is used. The AI world really deals with data, here lets consider the numerical values if we dig deeper the higher the numeric values higher is the computation power higher the usage of memory. 

   * For example: 17*6 = 102 , an easy computation whereas 17772324*6 = needs the energy/power to provide the result. 

Same here applies for memory & computation & networks, larger the data larger the work needed. Lets move to DNN's, here we might face a problem of gradient explosion because of larger values, so to avoid this risks we will scale the data and get them in a range of (-1,1) which is a standard range. 

In DNN's, we perform Batch-Normalization which meant scaling & shifting the values, as there are other normalization techniques such as Layer Normalization, Group Normalization, Instance Normalization.
