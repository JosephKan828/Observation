# Observation

## Project strucuture
```md
├── Code
│   ├── CloudSat
│   │   ├── LRF
│   │   └── Process
│   └── IMERG
├── Figure
│   ├── IMERG_corr
│   ├── OLR_corr
│   └── Vary_composite
└── Files
    ├── IMERG
    │   └── Corr
    └── OLR
        └── Corr
```

## Objective

This project aims to apply composite analysis and correlation analysis on Kelvin wave (KW) and Madden-Julia Oscillation (MJO). 

## Dataset

1. Convection proxies dataset
    KWs and MJO events are detected through their convective activity, and is represented by two proxies, including precipitation and outgoing longwave radiation (OLR).
    
    * precipiation
        Precipitation data from IMERG dataset is employed as a proxy of convective activity. The sampling frequency is daily, horizontal resolution is $0.625^\circ \times 0.5^\circ$. 
    * OLR

2. 