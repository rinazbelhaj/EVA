# What are Channels and Kernels ?

## Filters / Kernels:
**Filters** are used in **image processing** for blurring, sharpening, embossing, edge detection and more. In the initial layers of CNN, filters when convolved across an image captures low level features like vertical, horizontal and diagonal edges. Filter in the subsequent layers will be able to detect abstract and complex features. Each filter is a matrix of numbers corresponding to a feature that the filter is looking for. In CNN, filters are learned during training process. Each filter in a layer is randomly initialized to some distribution and hence will be able to detect separate features while training.

![alt+text](https://github.com/rinazbelhaj/EIP/blob/master/kernel.png?raw=true "Kernel")

## Channels:
**Channels** are features extracted by convolving **kernels** over an input image. Number of channels or features extracted equals the number of kernels used. More number of features extracted will help us in identifying images accurately. The channels are kept separate to avoid loss of information. Each channel carries specific set of information that will be merged together in final layers of neural network to correctly understand an image.

# Why should we only (well mostly) use 3x3 Kernels?

1.  3x3 kernels are least number of parameters [9 Parameters]
2.  3x3 kernels are accelerated on GPUs and TPUs 

# How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 ?

We need to perform 3x3 convolution 99 times on the input image of size 199x199 to make it 1x1. ie, we need to have 99 hidden layers between input and output layers.

## Calculation:

**Input = 199**

**Output = 1**

**Filter Size = 3**

**No of layer required = (Input-Output)/(Filter_Size-1)**

                       = (199-1)/(3-1)
                     
                       = 198/2
                     
                       = 99
                     
## Steps:

<pre>Layer : 1     Input Size : 199 x 199 ---> Convolution : 3 x 3 ---> Output : 197 x 197
Layer : 2     Input Size : 197 x 197 ---> Convolution : 3 x 3 ---> Output : 195 x 195
Layer : 3     Input Size : 195 x 195 ---> Convolution : 3 x 3 ---> Output : 193 x 193
Layer : 4     Input Size : 193 x 193 ---> Convolution : 3 x 3 ---> Output : 191 x 191</pre>
<pre>Layer : 5     Input Size : 191 x 191 ---> Convolution : 3 x 3 ---> Output : 189 x 189</pre>
<pre>Layer : 6     Input Size : 189 x 189 ---> Convolution : 3 x 3 ---> Output : 187 x 187</pre>
<pre>Layer : 7     Input Size : 187 x 187 ---> Convolution : 3 x 3 ---> Output : 185 x 185</pre>
<pre>Layer : 8     Input Size : 185 x 185 ---> Convolution : 3 x 3 ---> Output : 183 x 183</pre>
<pre>Layer : 9     Input Size : 183 x 183 ---> Convolution : 3 x 3 ---> Output : 181 x 181</pre>
<pre>Layer : 10     Input Size : 181 x 181 ---> Convolution : 3 x 3 ---> Output : 179 x 179</pre>
<pre>Layer : 11     Input Size : 179 x 179 ---> Convolution : 3 x 3 ---> Output : 177 x 177</pre>
<pre>Layer : 12     Input Size : 177 x 177 ---> Convolution : 3 x 3 ---> Output : 175 x 175</pre>
<pre>Layer : 13     Input Size : 175 x 175 ---> Convolution : 3 x 3 ---> Output : 173 x 173</pre>
<pre>Layer : 14     Input Size : 173 x 173 ---> Convolution : 3 x 3 ---> Output : 171 x 171</pre>
<pre>Layer : 15     Input Size : 171 x 171 ---> Convolution : 3 x 3 ---> Output : 169 x 169</pre>
<pre>Layer : 16     Input Size : 169 x 169 ---> Convolution : 3 x 3 ---> Output : 167 x 167</pre>
<pre>Layer : 17     Input Size : 167 x 167 ---> Convolution : 3 x 3 ---> Output : 165 x 165</pre>
<pre>Layer : 18     Input Size : 165 x 165 ---> Convolution : 3 x 3 ---> Output : 163 x 163</pre>
<pre>Layer : 19     Input Size : 163 x 163 ---> Convolution : 3 x 3 ---> Output : 161 x 161</pre>
<pre>Layer : 20     Input Size : 161 x 161 ---> Convolution : 3 x 3 ---> Output : 159 x 159</pre>
<pre>Layer : 21     Input Size : 159 x 159 ---> Convolution : 3 x 3 ---> Output : 157 x 157</pre>
<pre>Layer : 22     Input Size : 157 x 157 ---> Convolution : 3 x 3 ---> Output : 155 x 155</pre>
<pre>Layer : 23     Input Size : 155 x 155 ---> Convolution : 3 x 3 ---> Output : 153 x 153</pre>
<pre>Layer : 24     Input Size : 153 x 153 ---> Convolution : 3 x 3 ---> Output : 151 x 151</pre>
<pre>Layer : 25     Input Size : 151 x 151 ---> Convolution : 3 x 3 ---> Output : 149 x 149</pre>
<pre>Layer : 26     Input Size : 149 x 149 ---> Convolution : 3 x 3 ---> Output : 147 x 147</pre>
<pre>Layer : 27     Input Size : 147 x 147 ---> Convolution : 3 x 3 ---> Output : 145 x 145</pre>
<pre>Layer : 28     Input Size : 145 x 145 ---> Convolution : 3 x 3 ---> Output : 143 x 143</pre>
<pre>Layer : 29     Input Size : 143 x 143 ---> Convolution : 3 x 3 ---> Output : 141 x 141</pre>
<pre>Layer : 30     Input Size : 141 x 141 ---> Convolution : 3 x 3 ---> Output : 139 x 139</pre>
<pre>Layer : 31     Input Size : 139 x 139 ---> Convolution : 3 x 3 ---> Output : 137 x 137</pre>
<pre>Layer : 32     Input Size : 137 x 137 ---> Convolution : 3 x 3 ---> Output : 135 x 135</pre>
<pre>Layer : 33     Input Size : 135 x 135 ---> Convolution : 3 x 3 ---> Output : 133 x 133</pre>
<pre>Layer : 34     Input Size : 133 x 133 ---> Convolution : 3 x 3 ---> Output : 131 x 131</pre>
<pre>Layer : 35     Input Size : 131 x 131 ---> Convolution : 3 x 3 ---> Output : 129 x 129</pre>
<pre>Layer : 36     Input Size : 129 x 129 ---> Convolution : 3 x 3 ---> Output : 127 x 127</pre>
<pre>Layer : 37     Input Size : 127 x 127 ---> Convolution : 3 x 3 ---> Output : 125 x 125</pre>
<pre>Layer : 38     Input Size : 125 x 125 ---> Convolution : 3 x 3 ---> Output : 123 x 123</pre>
<pre>Layer : 39     Input Size : 123 x 123 ---> Convolution : 3 x 3 ---> Output : 121 x 121</pre>
<pre>Layer : 40     Input Size : 121 x 121 ---> Convolution : 3 x 3 ---> Output : 119 x 119</pre>
<pre>Layer : 41     Input Size : 119 x 119 ---> Convolution : 3 x 3 ---> Output : 117 x 117</pre>
<pre>Layer : 42     Input Size : 117 x 117 ---> Convolution : 3 x 3 ---> Output : 115 x 115</pre>
<pre>Layer : 43     Input Size : 115 x 115 ---> Convolution : 3 x 3 ---> Output : 113 x 113</pre>
<pre>Layer : 44     Input Size : 113 x 113 ---> Convolution : 3 x 3 ---> Output : 111 x 111</pre>
<pre>Layer : 45     Input Size : 111 x 111 ---> Convolution : 3 x 3 ---> Output : 109 x 109</pre>
<pre>Layer : 46     Input Size : 109 x 109 ---> Convolution : 3 x 3 ---> Output : 107 x 107</pre>
<pre>Layer : 47     Input Size : 107 x 107 ---> Convolution : 3 x 3 ---> Output : 105 x 105</pre>
<pre>Layer : 48     Input Size : 105 x 105 ---> Convolution : 3 x 3 ---> Output : 103 x 103</pre>
<pre>Layer : 49     Input Size : 103 x 103 ---> Convolution : 3 x 3 ---> Output : 101 x 101</pre>
<pre>Layer : 50     Input Size : 101 x 101 ---> Convolution : 3 x 3 ---> Output : 99 x 99</pre>
<pre>Layer : 51     Input Size : 99 x 99 ---> Convolution : 3 x 3 ---> Output : 97 x 97</pre>
<pre>Layer : 52     Input Size : 97 x 97 ---> Convolution : 3 x 3 ---> Output : 95 x 95</pre>
<pre>Layer : 53     Input Size : 95 x 95 ---> Convolution : 3 x 3 ---> Output : 93 x 93</pre>
<pre>Layer : 54     Input Size : 93 x 93 ---> Convolution : 3 x 3 ---> Output : 91 x 91</pre>
<pre>Layer : 55     Input Size : 91 x 91 ---> Convolution : 3 x 3 ---> Output : 89 x 89</pre>
<pre>Layer : 56     Input Size : 89 x 89 ---> Convolution : 3 x 3 ---> Output : 87 x 87</pre>
<pre>Layer : 57     Input Size : 87 x 87 ---> Convolution : 3 x 3 ---> Output : 85 x 85</pre>
<pre>Layer : 58     Input Size : 85 x 85 ---> Convolution : 3 x 3 ---> Output : 83 x 83</pre>
<pre>Layer : 59     Input Size : 83 x 83 ---> Convolution : 3 x 3 ---> Output : 81 x 81</pre>
<pre>Layer : 60     Input Size : 81 x 81 ---> Convolution : 3 x 3 ---> Output : 79 x 79</pre>
<pre>Layer : 61     Input Size : 79 x 79 ---> Convolution : 3 x 3 ---> Output : 77 x 77</pre>
<pre>Layer : 62     Input Size : 77 x 77 ---> Convolution : 3 x 3 ---> Output : 75 x 75</pre>
<pre>Layer : 63     Input Size : 75 x 75 ---> Convolution : 3 x 3 ---> Output : 73 x 73</pre>
<pre>Layer : 64     Input Size : 73 x 73 ---> Convolution : 3 x 3 ---> Output : 71 x 71</pre>
<pre>Layer : 65     Input Size : 71 x 71 ---> Convolution : 3 x 3 ---> Output : 69 x 69</pre>
<pre>Layer : 66     Input Size : 69 x 69 ---> Convolution : 3 x 3 ---> Output : 67 x 67</pre>
<pre>Layer : 67     Input Size : 67 x 67 ---> Convolution : 3 x 3 ---> Output : 65 x 65</pre>
<pre>Layer : 68     Input Size : 65 x 65 ---> Convolution : 3 x 3 ---> Output : 63 x 63</pre>
<pre>Layer : 69     Input Size : 63 x 63 ---> Convolution : 3 x 3 ---> Output : 61 x 61</pre>
<pre>Layer : 70     Input Size : 61 x 61 ---> Convolution : 3 x 3 ---> Output : 59 x 59</pre>
<pre>Layer : 71     Input Size : 59 x 59 ---> Convolution : 3 x 3 ---> Output : 57 x 57</pre>
<pre>Layer : 72     Input Size : 57 x 57 ---> Convolution : 3 x 3 ---> Output : 55 x 55</pre>
<pre>Layer : 73     Input Size : 55 x 55 ---> Convolution : 3 x 3 ---> Output : 53 x 53</pre>
<pre>Layer : 74     Input Size : 53 x 53 ---> Convolution : 3 x 3 ---> Output : 51 x 51</pre>
<pre>Layer : 75     Input Size : 51 x 51 ---> Convolution : 3 x 3 ---> Output : 49 x 49</pre>
<pre>Layer : 76     Input Size : 49 x 49 ---> Convolution : 3 x 3 ---> Output : 47 x 47</pre>
<pre>Layer : 77     Input Size : 47 x 47 ---> Convolution : 3 x 3 ---> Output : 45 x 45</pre>
<pre>Layer : 78     Input Size : 45 x 45 ---> Convolution : 3 x 3 ---> Output : 43 x 43</pre>
<pre>Layer : 79     Input Size : 43 x 43 ---> Convolution : 3 x 3 ---> Output : 41 x 41</pre>
<pre>Layer : 80     Input Size : 41 x 41 ---> Convolution : 3 x 3 ---> Output : 39 x 39</pre>
<pre>Layer : 81     Input Size : 39 x 39 ---> Convolution : 3 x 3 ---> Output : 37 x 37</pre>
<pre>Layer : 82     Input Size : 37 x 37 ---> Convolution : 3 x 3 ---> Output : 35 x 35</pre>
<pre>Layer : 83     Input Size : 35 x 35 ---> Convolution : 3 x 3 ---> Output : 33 x 33</pre>
<pre>Layer : 84     Input Size : 33 x 33 ---> Convolution : 3 x 3 ---> Output : 31 x 31</pre>
<pre>Layer : 85     Input Size : 31 x 31 ---> Convolution : 3 x 3 ---> Output : 29 x 29</pre>
<pre>Layer : 86     Input Size : 29 x 29 ---> Convolution : 3 x 3 ---> Output : 27 x 27</pre>
<pre>Layer : 87     Input Size : 27 x 27 ---> Convolution : 3 x 3 ---> Output : 25 x 25</pre>
<pre>Layer : 88     Input Size : 25 x 25 ---> Convolution : 3 x 3 ---> Output : 23 x 23</pre>
<pre>Layer : 89     Input Size : 23 x 23 ---> Convolution : 3 x 3 ---> Output : 21 x 21</pre>
<pre>Layer : 90     Input Size : 21 x 21 ---> Convolution : 3 x 3 ---> Output : 19 x 19</pre>
<pre>Layer : 91     Input Size : 19 x 19 ---> Convolution : 3 x 3 ---> Output : 17 x 17</pre>
<pre>Layer : 92     Input Size : 17 x 17 ---> Convolution : 3 x 3 ---> Output : 15 x 15</pre>
<pre>Layer : 93     Input Size : 15 x 15 ---> Convolution : 3 x 3 ---> Output : 13 x 13</pre>
<pre>Layer : 94     Input Size : 13 x 13 ---> Convolution : 3 x 3 ---> Output : 11 x 11</pre>
<pre>Layer : 95     Input Size : 11 x 11 ---> Convolution : 3 x 3 ---> Output : 9 x 9</pre>
<pre>Layer : 96     Input Size : 9 x 9 ---> Convolution : 3 x 3 ---> Output : 7 x 7</pre>
<pre>Layer : 97     Input Size : 7 x 7 ---> Convolution : 3 x 3 ---> Output : 5 x 5</pre>
<pre>Layer : 98     Input Size : 5 x 5 ---> Convolution : 3 x 3 ---> Output : 3 x 3</pre>
<pre>Layer : 99     Input Size : 3 x 3 ---> Convolution : 3 x 3 ---> Output : 1 x 1</pre>
