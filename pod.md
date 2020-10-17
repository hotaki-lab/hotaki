## Reliability Analysis via Probability of Detection (POD) Curve:

### 1. What is a POD Curve:

The Department of Defense developed a guide book (MIL-HDBK-1823A) for reliability assessment of nondestructive evaluation systems. The main focus of this guide book was to assess reliability of NDE systems and sensors by POD analysis as a function of flaw size. POD curves are not used to assess the smallest crack an NDT method can detect, but rather to assess the largest crack a method may miss. According to Volker, it has been determined that an NDE technique does not have a specific POD because POD curve does not depend only on physics of the technique, but also other factors as discussed in a study conducted by Forli (56).


### 2. Assumptions and Approach (Using R):

Two statistical approaches for constructing a POD curve were considered in the guide book. One approach was based on signal response and another was based on **binary (hit/miss)** data. This book specifically discussed the application of these POD approaches on UT, eddy current (ET), MT, and fluorescent penetrant (PT) tests.
The MH-1823 algorithm is written in R and calculates the POD using four link functions to plot sizes within (−∞ < x < ∞) and POD within (0 < y < 1). These functions are the **logit** (logistic or **log-odds** function), **probit** (inverse normal function), the **complementary log-log** function (often called Weibull by engineers), and **loglog** function. These link functions are used to connect the values at 0 (missed flaws) to the values at 1 (hit data) and to estimate a POD for the system. These functions are as below:


<img src="images/loglog.JPG?raw=true"/>


<img src="images/utpod.JPG?raw=true"/>


<img src="images/RT.JPG?raw=true"/>


<img src="images/RT-UT.JPG?raw=true"/>


### 3. [Full Research Document Can be Accessed Here](/pdf/research.pdf)


**Project Location:** [The Sherman Minton Bridge is a two span tied arch bridge spanning the Ohio River and connecting Indiana and Kentucky states.](https://www.google.com/maps/place/Sherman+Minton+Bridge/@38.2787315,-85.8244487,17z/data=!3m1!4b1!4m5!3m4!1s0x88696cf146f65fed:0xec17b638d8fc4378!8m2!3d38.2787315!4d-85.82226) 
