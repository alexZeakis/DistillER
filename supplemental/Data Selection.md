# Data Selection

## Positive Ratio of different Strategies
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="1">Random</th>
      <th colspan="1">Blocking Max</th>
      <th colspan="1">Blocking Top2</th>
      <th colspan="1">Blocking Max (p=0.85)</th>
      <th colspan="1">Blocking Top2 (p=0.85)</th>
      <th colspan="1">Clustering Hierarchical</th>
      <th colspan="1">Clustering KMeans</th>
      <th colspan="1">Clustering Hierarchical (p=0.85)</th>
      <th colspan="1">Clustering KMeans (p=0.85)</th>
      <th colspan="1">Clustering Hierarchical (b=6)</th>
      <th colspan="1">Clustering KMeans (b=6)</th>
      <th colspan="1">Sampled</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>D2</td><td>0.96</td><td>0.94</td><td>0.95</td><td>0.97</td><td>0.96</td><td>0.99</td><td>0.94</td><td>0.99</td><td>0.94</td><td>0.96</td><td>0.97</td><td>0.75</td></tr>
    <tr><td>D3</td><td>0.75</td><td>0.58</td><td>0.61</td><td>0.61</td><td>0.64</td><td>0.72</td><td>0.72</td><td>0.74</td><td>0.75</td><td>0.74</td><td>0.70</td><td>0.75</td></tr>
    <tr><td>D4</td><td>0.98</td><td>0.90</td><td>0.93</td><td>0.93</td><td>0.93</td><td>0.98</td><td>0.99</td><td>0.97</td><td>0.99</td><td>0.97</td><td>0.99</td><td>0.75</td></tr>
    <tr><td>D5</td><td>0.42</td><td>0.74</td><td>0.67</td><td>0.84</td><td>0.76</td><td>0.44</td><td>0.44</td><td>0.44</td><td>0.44</td><td>0.43</td><td>0.44</td><td>0.75</td></tr>
    <tr><td>D6</td><td>0.17</td><td>0.46</td><td>0.44</td><td>0.50</td><td>0.50</td><td>0.17</td><td>0.14</td><td>0.16</td><td>0.12</td><td>0.12</td><td>0.14</td><td>0.75</td></tr>
    <tr><td>D7</td><td>0.20</td><td>0.43</td><td>0.42</td><td>0.50</td><td>0.48</td><td>0.23</td><td>0.24</td><td>0.23</td><td>0.23</td><td>0.18</td><td>0.24</td><td>0.75</td></tr>
    <tr><td>D8</td><td>0.34</td><td>0.50</td><td>0.46</td><td>0.56</td><td>0.52</td><td>0.38</td><td>0.35</td><td>0.38</td><td>0.38</td><td>0.31</td><td>0.29</td><td>0.75</td></tr>
    <tr><td>D9</td><td>0.94</td><td>0.82</td><td>0.83</td><td>0.87</td><td>0.87</td><td>0.91</td><td>0.88</td><td>0.91</td><td>0.88</td><td>0.87</td><td>0.89</td><td>0.75</td></tr>
    <tr><td><b>Mean</b></td><td>0.59</td><td>0.67</td><td>0.67</td><td>0.72</td><td>0.71</td><td>0.60</td><td>0.59</td><td>0.60</td><td>0.59</td><td>0.57</td><td>0.58</td><td>0.75</td></tr>
  </tbody>
</table>



## Evaluation of different Strategies
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Random</th>
      <th colspan="2">Blocking Max</th>
      <th colspan="2">Blocking Top2</th>
      <th colspan="2">Blocking Max (p=0.85)</th>
      <th colspan="2">Blocking Top2 (p=0.85)</th>
      <th colspan="2">Clustering Hierarchical</th>
      <th colspan="2">Clustering KMeans</th>
      <th colspan="2">Clustering Hierarchical (p=0.85)</th>
      <th colspan="2">Clustering KMeans (p=0.85)</th>
      <th colspan="2">Clustering Hierarchical (b=6)</th>
      <th colspan="2">Clustering KMeans (b=6)</th>
      <th colspan="2">Sampled</th>
    </tr>
    <tr>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
      <th>llama_8</th><th>qwen_14</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>D2</td><td>0.89</td><td>0.94</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.91</td><td>0.96</td><td>0.94</td><td>0.92</td><td>0.89</td><td>0.88</td><td>0.90</td><td>0.87</td><td>0.90</td><td>0.90</td><td>0.89</td><td>0.88</td><td>0.92</td><td>0.88</td><td>0.91</td><td>0.90</td><td>0.95</td><td>0.68</td><td>0.86</td></tr>
    <tr><td>D3</td><td>0.58</td><td>0.69</td><td>0.57</td><td>0.64</td><td>0.54</td><td>0.61</td><td>0.60</td><td>0.63</td><td>0.56</td><td>0.60</td><td>0.46</td><td>0.59</td><td>0.51</td><td>0.58</td><td>0.51</td><td>0.58</td><td>0.56</td><td>0.59</td><td>0.58</td><td>0.60</td><td>0.60</td><td>0.59</td><td>0.61</td><td>0.69</td></tr>
    <tr><td>D4</td><td>0.96</td><td>0.97</td><td>0.86</td><td>0.93</td><td>0.88</td><td>0.95</td><td>0.91</td><td>0.96</td><td>0.89</td><td>0.96</td><td>0.96</td><td>0.97</td><td>0.96</td><td>0.99</td><td>0.94</td><td>0.97</td><td>0.95</td><td>0.99</td><td>0.92</td><td>0.97</td><td>0.97</td><td>1.00</td><td>0.74</td><td>0.81</td></tr>
    <tr><td>D5</td><td>0.61</td><td>0.78</td><td>0.87</td><td>0.98</td><td>0.81</td><td>0.94</td><td>0.93</td><td>0.99</td><td>0.86</td><td>0.93</td><td>0.69</td><td>0.86</td><td>0.67</td><td>0.83</td><td>0.62</td><td>0.87</td><td>0.64</td><td>0.85</td><td>0.64</td><td>0.84</td><td>0.63</td><td>0.83</td><td>0.85</td><td>0.94</td></tr>
    <tr><td>D6</td><td>0.43</td><td>0.81</td><td>0.61</td><td>0.72</td><td>0.57</td><td>0.71</td><td>0.55</td><td>0.67</td><td>0.54</td><td>0.67</td><td>0.44</td><td>0.83</td><td>0.39</td><td>0.80</td><td>0.45</td><td>0.83</td><td>0.43</td><td>0.81</td><td>0.42</td><td>0.82</td><td>0.44</td><td>0.81</td><td>0.77</td><td>0.82</td></tr>
    <tr><td>D7</td><td>0.34</td><td>0.66</td><td>0.50</td><td>0.63</td><td>0.48</td><td>0.63</td><td>0.54</td><td>0.61</td><td>0.52</td><td>0.59</td><td>0.36</td><td>0.70</td><td>0.40</td><td>0.70</td><td>0.37</td><td>0.70</td><td>0.37</td><td>0.72</td><td>0.37</td><td>0.72</td><td>0.38</td><td>0.71</td><td>0.74</td><td>0.88</td></tr>
    <tr><td>D8</td><td>0.36</td><td>0.67</td><td>0.59</td><td>0.77</td><td>0.55</td><td>0.75</td><td>0.63</td><td>0.76</td><td>0.60</td><td>0.72</td><td>0.44</td><td>0.65</td><td>0.38</td><td>0.65</td><td>0.56</td><td>0.64</td><td>0.61</td><td>0.65</td><td>0.52</td><td>0.70</td><td>0.53</td><td>0.64</td><td>0.64</td><td>0.82</td></tr>
    <tr><td>D9</td><td>0.90</td><td>0.96</td><td>0.83</td><td>0.91</td><td>0.82</td><td>0.88</td><td>0.89</td><td>0.94</td><td>0.86</td><td>0.92</td><td>0.85</td><td>0.93</td><td>0.83</td><td>0.92</td><td>0.84</td><td>0.93</td><td>0.83</td><td>0.91</td><td>0.82</td><td>0.91</td><td>0.80</td><td>0.90</td><td>0.73</td><td>0.87</td></tr>
    <tr><td><b>Mean</b></td><td>0.63</td><td>0.81</td><td>0.72</td><td>0.81</td><td>0.70</td><td>0.80</td><td>0.75</td><td>0.81</td><td>0.72</td><td>0.79</td><td>0.63</td><td>0.80</td><td>0.63</td><td>0.80</td><td>0.65</td><td>0.80</td><td>0.66</td><td>0.81</td><td>0.64</td><td>0.81</td><td>0.66</td><td>0.80</td><td>0.72</td><td>0.84</td></tr>
  </tbody>
</table>

