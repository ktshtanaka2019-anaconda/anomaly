This file is an experimental programme for anomaly detection using electricity data from Tokyo Electric Power Grid Inc. in Japan and data from the Tokyo Meteorological Observatory of the Japan Meteorological Agency.

The respective data files can be downloaded from the following locations:

* Tokyo Electric Power Grid data
https://www.tepco.co.jp/forecast/html/juyo-j.html

* Japan Meteorological Agency Weather Data
https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=44&block_no=47662&year=2026&month=2&day=14&view=

The purpose of this programme is to conduct experiments using pymod's AutoEncoder to detect anomalies between meteorological data and power supply.

Please note that the recommended environment is Anaconda's Anaconda environment.

Required environment setup.

*pymod
*pytorch
*pandas
*sklearn
