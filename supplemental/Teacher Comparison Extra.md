# Teacher Comparison

## Precision
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Large LLMs</th>
      <th colspan="2">Small LLMs</th>
      <th colspan="2">Small Encoders (32)</th>
      <th colspan="2">Large Encoders (70)</th>
      <th rowspan="2">Multi-Teacher</th>
      <th rowspan="2">Any</th>
    </tr>
    <tr>
      <th>Llama-70b</th>
      <th>Qwen-32b</th>
      <th>Llama-8b</th>
      <th>Qwen-14b</th>
      <th>SMiniLM-32</th>
      <th>RoBERTa-32</th>
      <th>SMiniLM-70</th>
      <th>RoBERTa-70</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>D2</td><td>0.97</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.36</td><td>0.59</td><td>0.39</td><td>0.47</td><td>0.95</td><td>0.85</td></tr>
    <tr><td>D3</td><td>0.66</td><td>0.65</td><td>0.57</td><td>0.64</td><td>0.38</td><td>0.45</td><td>0.33</td><td>0.48</td><td>0.67</td><td>0.56</td></tr>
    <tr><td>D4</td><td>0.94</td><td>0.96</td><td>0.86</td><td>0.93</td><td>0.71</td><td>0.77</td><td>0.74</td><td>0.75</td><td>0.93</td><td>0.85</td></tr>
    <tr><td>D5</td><td>0.98</td><td>0.99</td><td>0.87</td><td>0.98</td><td>0.83</td><td>0.84</td><td>0.84</td><td>0.85</td><td>0.98</td><td>0.87</td></tr>
    <tr><td>D6</td><td>0.71</td><td>0.73</td><td>0.61</td><td>0.72</td><td>0.64</td><td>0.65</td><td>0.64</td><td>0.63</td><td>0.73</td><td>0.60</td></tr>
    <tr><td>D7</td><td>0.66</td><td>0.63</td><td>0.50</td><td>0.63</td><td>0.62</td><td>0.64</td><td>0.66</td><td>0.66</td><td>0.65</td><td>0.49</td></tr>
    <tr><td>D8</td><td>0.74</td><td>0.84</td><td>0.59</td><td>0.77</td><td>0.59</td><td>0.70</td><td>0.57</td><td>0.65</td><td>0.82</td><td>0.59</td></tr>
    <tr><td>D9</td><td>0.89</td><td>0.96</td><td>0.83</td><td>0.91</td><td>0.73</td><td>0.78</td><td>0.74</td><td>0.79</td><td>0.93</td><td>0.80</td></tr>
    <tr><td><b>Mean</b></td><td>0.82</td><td>0.83</td><td>0.72</td><td>0.81</td><td>0.61</td><td>0.68</td><td>0.61</td><td>0.66</td><td>0.83</td><td>0.70</td></tr>
  </tbody>
</table>



## Recall
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Large LLMs</th>
      <th colspan="2">Small LLMs</th>
      <th colspan="2">Small Encoders (32)</th>
      <th colspan="2">Large Encoders (70)</th>
      <th rowspan="2">Multi-Teacher</th>
      <th rowspan="2">Any</th>
    </tr>
    <tr>
      <th>Llama-70b</th>
      <th>Qwen-32b</th>
      <th>Llama-8b</th>
      <th>Qwen-14b</th>
      <th>SMiniLM-32</th>
      <th>RoBERTa-32</th>
      <th>SMiniLM-70</th>
      <th>RoBERTa-70</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>D2</td><td>0.97</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.48</td><td>0.63</td><td>0.60</td><td>0.54</td><td>0.95</td><td>0.97</td></tr>
    <tr><td>D3</td><td>0.66</td><td>0.65</td><td>0.57</td><td>0.64</td><td>0.49</td><td>0.47</td><td>0.43</td><td>0.51</td><td>0.67</td><td>0.72</td></tr>
    <tr><td>D4</td><td>0.94</td><td>0.96</td><td>0.86</td><td>0.93</td><td>0.80</td><td>0.79</td><td>0.79</td><td>0.78</td><td>0.93</td><td>0.97</td></tr>
    <tr><td>D5</td><td>0.98</td><td>0.99</td><td>0.87</td><td>0.98</td><td>0.85</td><td>0.86</td><td>0.85</td><td>0.85</td><td>0.98</td><td>0.99</td></tr>
    <tr><td>D6</td><td>0.71</td><td>0.73</td><td>0.61</td><td>0.72</td><td>0.67</td><td>0.66</td><td>0.67</td><td>0.67</td><td>0.73</td><td>0.76</td></tr>
    <tr><td>D7</td><td>0.66</td><td>0.63</td><td>0.50</td><td>0.63</td><td>0.63</td><td>0.64</td><td>0.67</td><td>0.66</td><td>0.65</td><td>0.76</td></tr>
    <tr><td>D8</td><td>0.74</td><td>0.84</td><td>0.59</td><td>0.77</td><td>0.71</td><td>0.75</td><td>0.71</td><td>0.71</td><td>0.82</td><td>0.87</td></tr>
    <tr><td>D9</td><td>0.89</td><td>0.96</td><td>0.83</td><td>0.91</td><td>0.79</td><td>0.80</td><td>0.77</td><td>0.81</td><td>0.93</td><td>0.97</td></tr>
    <tr><td><b>Mean</b></td><td>0.82</td><td>0.83</td><td>0.72</td><td>0.81</td><td>0.68</td><td>0.70</td><td>0.69</td><td>0.69</td><td>0.83</td><td>0.88</td></tr>
  </tbody>
</table>

