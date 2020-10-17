{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<a href=\"https://www.bigdatauniversity.com\"><img src = \"https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png\" width = 400, align = \"center\"></a>\n",
    "\n",
    "# <center>Simple Linear Regression</center>\n",
    "\n",
    "\n",
    "#### About this Notebook\n",
    "In this notebook, we learn how to use scikit-learn to implement simple linear regression. We download a dataset that is related to fuel consumption and Carbon dioxide emission of cars. Then, we split our data into training and test sets, create a model using training set, Evaluate your model using test set, and finally use model to predict unknown value\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Importing Needed packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import pylab as pl\n",
    "import numpy as np\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Downloading Data\n",
    "To download the data, we will use !wget to download it from IBM Object Storage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"FuelConsumption.csv\")\n",
    "\n",
    "#!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__Did you know?__ When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "\n",
    "## Understanding the Data\n",
    "\n",
    "### `FuelConsumption.csv`:\n",
    "We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)\n",
    "\n",
    "- **MODELYEAR** e.g. 2014\n",
    "- **MAKE** e.g. Acura\n",
    "- **MODEL** e.g. ILX\n",
    "- **VEHICLE CLASS** e.g. SUV\n",
    "- **ENGINE SIZE** e.g. 4.7\n",
    "- **CYLINDERS** e.g 6\n",
    "- **TRANSMISSION** e.g. A6\n",
    "- **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9\n",
    "- **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9\n",
    "- **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2\n",
    "- **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Reading the data in"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MODELYEAR</th>\n",
       "      <th>MAKE</th>\n",
       "      <th>MODEL</th>\n",
       "      <th>VEHICLECLASS</th>\n",
       "      <th>ENGINESIZE</th>\n",
       "      <th>CYLINDERS</th>\n",
       "      <th>TRANSMISSION</th>\n",
       "      <th>FUELTYPE</th>\n",
       "      <th>FUELCONSUMPTION_CITY</th>\n",
       "      <th>FUELCONSUMPTION_HWY</th>\n",
       "      <th>FUELCONSUMPTION_COMB</th>\n",
       "      <th>FUELCONSUMPTION_COMB_MPG</th>\n",
       "      <th>CO2EMISSIONS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>ILX</td>\n",
       "      <td>COMPACT</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "      <td>AS5</td>\n",
       "      <td>Z</td>\n",
       "      <td>9.9</td>\n",
       "      <td>6.7</td>\n",
       "      <td>8.5</td>\n",
       "      <td>33</td>\n",
       "      <td>196</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>ILX</td>\n",
       "      <td>COMPACT</td>\n",
       "      <td>2.4</td>\n",
       "      <td>4</td>\n",
       "      <td>M6</td>\n",
       "      <td>Z</td>\n",
       "      <td>11.2</td>\n",
       "      <td>7.7</td>\n",
       "      <td>9.6</td>\n",
       "      <td>29</td>\n",
       "      <td>221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>ILX HYBRID</td>\n",
       "      <td>COMPACT</td>\n",
       "      <td>1.5</td>\n",
       "      <td>4</td>\n",
       "      <td>AV7</td>\n",
       "      <td>Z</td>\n",
       "      <td>6.0</td>\n",
       "      <td>5.8</td>\n",
       "      <td>5.9</td>\n",
       "      <td>48</td>\n",
       "      <td>136</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>MDX 4WD</td>\n",
       "      <td>SUV - SMALL</td>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>AS6</td>\n",
       "      <td>Z</td>\n",
       "      <td>12.7</td>\n",
       "      <td>9.1</td>\n",
       "      <td>11.1</td>\n",
       "      <td>25</td>\n",
       "      <td>255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2014</td>\n",
       "      <td>ACURA</td>\n",
       "      <td>RDX AWD</td>\n",
       "      <td>SUV - SMALL</td>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>AS6</td>\n",
       "      <td>Z</td>\n",
       "      <td>12.1</td>\n",
       "      <td>8.7</td>\n",
       "      <td>10.6</td>\n",
       "      <td>27</td>\n",
       "      <td>244</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   MODELYEAR   MAKE       MODEL VEHICLECLASS  ENGINESIZE  CYLINDERS  \\\n",
       "0       2014  ACURA         ILX      COMPACT         2.0          4   \n",
       "1       2014  ACURA         ILX      COMPACT         2.4          4   \n",
       "2       2014  ACURA  ILX HYBRID      COMPACT         1.5          4   \n",
       "3       2014  ACURA     MDX 4WD  SUV - SMALL         3.5          6   \n",
       "4       2014  ACURA     RDX AWD  SUV - SMALL         3.5          6   \n",
       "\n",
       "  TRANSMISSION FUELTYPE  FUELCONSUMPTION_CITY  FUELCONSUMPTION_HWY  \\\n",
       "0          AS5        Z                   9.9                  6.7   \n",
       "1           M6        Z                  11.2                  7.7   \n",
       "2          AV7        Z                   6.0                  5.8   \n",
       "3          AS6        Z                  12.7                  9.1   \n",
       "4          AS6        Z                  12.1                  8.7   \n",
       "\n",
       "   FUELCONSUMPTION_COMB  FUELCONSUMPTION_COMB_MPG  CO2EMISSIONS  \n",
       "0                   8.5                        33           196  \n",
       "1                   9.6                        29           221  \n",
       "2                   5.9                        48           136  \n",
       "3                  11.1                        25           255  \n",
       "4                  10.6                        27           244  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"FuelConsumption.csv\")\n",
    "\n",
    "# take a look at the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Data Exploration\n",
    "Lets first have a descriptive exploration on our data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MODELYEAR</th>\n",
       "      <th>ENGINESIZE</th>\n",
       "      <th>CYLINDERS</th>\n",
       "      <th>FUELCONSUMPTION_CITY</th>\n",
       "      <th>FUELCONSUMPTION_HWY</th>\n",
       "      <th>FUELCONSUMPTION_COMB</th>\n",
       "      <th>FUELCONSUMPTION_COMB_MPG</th>\n",
       "      <th>CO2EMISSIONS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1067.0</td>\n",
       "      <td>1067.000000</td>\n",
       "      <td>1067.000000</td>\n",
       "      <td>1067.000000</td>\n",
       "      <td>1067.000000</td>\n",
       "      <td>1067.000000</td>\n",
       "      <td>1067.000000</td>\n",
       "      <td>1067.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2014.0</td>\n",
       "      <td>3.346298</td>\n",
       "      <td>5.794752</td>\n",
       "      <td>13.296532</td>\n",
       "      <td>9.474602</td>\n",
       "      <td>11.580881</td>\n",
       "      <td>26.441425</td>\n",
       "      <td>256.228679</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.415895</td>\n",
       "      <td>1.797447</td>\n",
       "      <td>4.101253</td>\n",
       "      <td>2.794510</td>\n",
       "      <td>3.485595</td>\n",
       "      <td>7.468702</td>\n",
       "      <td>63.372304</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2014.0</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.600000</td>\n",
       "      <td>4.900000</td>\n",
       "      <td>4.700000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>108.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2014.0</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>10.250000</td>\n",
       "      <td>7.500000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>207.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>2014.0</td>\n",
       "      <td>3.400000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>12.600000</td>\n",
       "      <td>8.800000</td>\n",
       "      <td>10.900000</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>251.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2014.0</td>\n",
       "      <td>4.300000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>15.550000</td>\n",
       "      <td>10.850000</td>\n",
       "      <td>13.350000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>294.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2014.0</td>\n",
       "      <td>8.400000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>30.200000</td>\n",
       "      <td>20.500000</td>\n",
       "      <td>25.800000</td>\n",
       "      <td>60.000000</td>\n",
       "      <td>488.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       MODELYEAR   ENGINESIZE    CYLINDERS  FUELCONSUMPTION_CITY  \\\n",
       "count     1067.0  1067.000000  1067.000000           1067.000000   \n",
       "mean      2014.0     3.346298     5.794752             13.296532   \n",
       "std          0.0     1.415895     1.797447              4.101253   \n",
       "min       2014.0     1.000000     3.000000              4.600000   \n",
       "25%       2014.0     2.000000     4.000000             10.250000   \n",
       "50%       2014.0     3.400000     6.000000             12.600000   \n",
       "75%       2014.0     4.300000     8.000000             15.550000   \n",
       "max       2014.0     8.400000    12.000000             30.200000   \n",
       "\n",
       "       FUELCONSUMPTION_HWY  FUELCONSUMPTION_COMB  FUELCONSUMPTION_COMB_MPG  \\\n",
       "count          1067.000000           1067.000000               1067.000000   \n",
       "mean              9.474602             11.580881                 26.441425   \n",
       "std               2.794510              3.485595                  7.468702   \n",
       "min               4.900000              4.700000                 11.000000   \n",
       "25%               7.500000              9.000000                 21.000000   \n",
       "50%               8.800000             10.900000                 26.000000   \n",
       "75%              10.850000             13.350000                 31.000000   \n",
       "max              20.500000             25.800000                 60.000000   \n",
       "\n",
       "       CO2EMISSIONS  \n",
       "count   1067.000000  \n",
       "mean     256.228679  \n",
       "std       63.372304  \n",
       "min      108.000000  \n",
       "25%      207.000000  \n",
       "50%      251.000000  \n",
       "75%      294.000000  \n",
       "max      488.000000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# summarize the data\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets select some features to explore more."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ENGINESIZE</th>\n",
       "      <th>CYLINDERS</th>\n",
       "      <th>FUELCONSUMPTION_COMB</th>\n",
       "      <th>CO2EMISSIONS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "      <td>8.5</td>\n",
       "      <td>196</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.4</td>\n",
       "      <td>4</td>\n",
       "      <td>9.6</td>\n",
       "      <td>221</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.5</td>\n",
       "      <td>4</td>\n",
       "      <td>5.9</td>\n",
       "      <td>136</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>11.1</td>\n",
       "      <td>255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>10.6</td>\n",
       "      <td>244</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>10.0</td>\n",
       "      <td>230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>3.5</td>\n",
       "      <td>6</td>\n",
       "      <td>10.1</td>\n",
       "      <td>232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>3.7</td>\n",
       "      <td>6</td>\n",
       "      <td>11.1</td>\n",
       "      <td>255</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>3.7</td>\n",
       "      <td>6</td>\n",
       "      <td>11.6</td>\n",
       "      <td>267</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ENGINESIZE  CYLINDERS  FUELCONSUMPTION_COMB  CO2EMISSIONS\n",
       "0         2.0          4                   8.5           196\n",
       "1         2.4          4                   9.6           221\n",
       "2         1.5          4                   5.9           136\n",
       "3         3.5          6                  11.1           255\n",
       "4         3.5          6                  10.6           244\n",
       "5         3.5          6                  10.0           230\n",
       "6         3.5          6                  10.1           232\n",
       "7         3.7          6                  11.1           255\n",
       "8         3.7          6                  11.6           267"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]\n",
    "cdf.head(9)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "we can plot each of these fearues:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 4 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]\n",
    "viz.hist()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, lets plot each of these features vs the Emission, to see how linear is their relation:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"FUELCONSUMPTION_COMB\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    },
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Practice\n",
    "plot __CYLINDER__ vs the Emission, to see how linear is their relation:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# write your code here\n",
    "\n",
    "plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')\n",
    "plt.xlabel('Cylinders')\n",
    "plt.ylabel('Emission')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1067"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Double-click __here__ for the solution.\n",
    "\n",
    "<!-- Your answer is below:\n",
    "    \n",
    "plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')\n",
    "plt.xlabel(\"Cylinders\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()\n",
    "\n",
    "-->"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Creating train and test dataset\n",
    "Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. \n",
    "This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. It is more realistic for real world problems.\n",
    "\n",
    "This means that we know the outcome of each data point in this dataset, making it great to test with! And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.\n",
    "\n",
    "Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. We create a mask to select random rows using __np.random.rand()__ function: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "msk = np.random.rand(len(df)) < 0.8\n",
    "train = cdf[msk]\n",
    "test = cdf[~msk]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Simple Regression Model\n",
    "Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the independent x in the dataset, and the dependent y by the linear approximation. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Train data distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Modeling\n",
    "Using sklearn package to model data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Coefficients:  [[39.61825581]]\n",
      "Intercept:  [123.34818189]\n"
     ]
    }
   ],
   "source": [
    "from sklearn import linear_model\n",
    "regr = linear_model.LinearRegression()\n",
    "train_x = np.asanyarray(train[['ENGINESIZE']])\n",
    "train_y = np.asanyarray(train[['CO2EMISSIONS']])\n",
    "regr.fit (train_x, train_y)\n",
    "# The coefficients\n",
    "print ('Coefficients: ', regr.coef_)\n",
    "print ('Intercept: ',regr.intercept_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As mentioned before, __Coefficient__ and __Intercept__ in the simple linear regression, are the parameters of the fit line. \n",
    "Given that it is a simple linear regression, with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, sklearn can estimate them directly from our data. \n",
    "Notice that all of the data must be available to traverse and calculate the parameters.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Plot outputs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "we can plot the fit line over the data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Emission')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')\n",
    "plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')\n",
    "plt.xlabel(\"Engine size\")\n",
    "plt.ylabel(\"Emission\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Evaluation\n",
    "we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.\n",
    "\n",
    "There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set: \n",
    "    - Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.\n",
    "    - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.\n",
    "    - Root Mean Squared Error (RMSE).\n",
    "    - R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean absolute error: 25.72\n",
      "Residual sum of squares (MSE): 1142.30\n",
      "R2-score: 0.66\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "\n",
    "test_x = np.asanyarray(test[['ENGINESIZE']])\n",
    "test_y = np.asanyarray(test[['CO2EMISSIONS']])\n",
    "test_y_ = regr.predict(test_x)\n",
    "\n",
    "print(\"Mean absolute error: %.2f\" % np.mean(np.absolute(test_y_ - test_y)))\n",
    "print(\"Residual sum of squares (MSE): %.2f\" % np.mean((test_y_ - test_y) ** 2))\n",
    "print(\"R2-score: %.2f\" % r2_score(test_y_ , test_y) )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Want to learn more?\n",
    "\n",
    "IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: [SPSS Modeler](http://cocl.us/ML0101EN-SPSSModeler).\n",
    "\n",
    "Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at [Watson Studio](https://cocl.us/ML0101EN_DSX)\n",
    "\n",
    "### Thanks for completing this lesson!\n",
    "\n",
    "Notebook created by: <a href = \"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a>\n",
    "\n",
    "<hr>\n",
    "Copyright &copy; 2018 [Cognitive Class](https://cocl.us/DX0108EN_CC). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).​"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
