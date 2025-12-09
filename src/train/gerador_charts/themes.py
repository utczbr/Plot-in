
# ===================================================================================
# == GLOBAL LISTS AND THEMES FOR REALISM ==
# ===================================================================================

SCIENTIFIC_Y_LABELS = [
    'Group', 'Treatment', 'Condition', 'Dose (mg/kg)', 'Dose (μg/mL)', 'Time Point (h)',
    'Time Point (min)', 'Concentration (μM)', 'Sample ID', 'Replicate', 'Batch',
    'Genotype', 'Phenotype', 'Cell Line', 'Species', 'Subject ID', 'Age (years)',
    'Sex', 'Location', 'Site', 'Visit Number', 'Passage Number', 'Culture Duration (hrs)',
    'Temperature (°C)', 'pH', 'Stimulus', 'Protocol', 'Experimental Phase', 'Arm',
    'Trial', 'Environmental Factor', 'Exposure Time (s)', 'Wavelength (nm)', 'Frequency (Hz)',
    'Observation Date', 'Collection Time', 'Severity', 'Treatment Arm', 'Sequence ID',
        'Response (a.u.)', 'Activity (%)', 'Time (s)', 'Time (min)', 'Time (h)',
    'Concentration (μM)', 'Concentration (mg/L)', 'Molarity (mM)', 'Fold Change',
    'Optical Density (OD)', 'OD600', 'Viability (%)', 'Survival Fraction',
    'Expression Level (a.u.)', 'Signal (RFU)', 'Luminescence (RLU)', 'Absorbance (A.U.)',
    'Rate (units/s)', 'Enzyme Activity (U/mL)', 'Binding (%)', 'Binding Affinity (Kd, nM)',
    'IC50 (nM)', 'EC50 (μM)', 'Titer (ng/mL)', 'Copy Number', 'Reads per million (RPM)',
    'TPM', 'RPKM', 'Mutation Frequency (%)', 'Allele Frequency (%)', 'Colony Forming Units (CFU/mL)',
    'Cell Count (cells/mL)', 'Growth Rate (doublings/hr)', 'pH', 'Temperature (°C)',
    'Pressure (kPa)', 'Intensity (a.u.)', 'Wavelength (nm)', 'Density (g/mL)',
    'Mass (mg)', 'Volume (mL)', 'Torque (Nm)', 'Force (N)', 'Voltage (mV)',
    'Current (µA)', 'Signal-to-Noise Ratio', '% Inhibition', 'Normalized Score',
    'Fluorescence (AU)', 'FRET Ratio', 'Anisotropy (r)', 'Quantum Yield (%)',
    'Permeability (cm/s)', 'Clearance (mL/min)', 'Half-life (t1/2, h)', 'AUC (ng·h/mL)',
    'Cmax (ng/mL)', 'Bioavailability (%)', 'Protein Binding (%)', 'Recovery (%)',
    'Purity (%)', 'Yield (%)', 'Conversion (%)', 'Selectivity (%)',
    'Retention Time (min)', 'Peak Area (mAU·s)', 'Resolution (Rs)', 'Theoretical Plates (N)',
    'Ct Value', 'ΔΔCt', 'Melting Temperature (Tm, °C)', 'Amplification Efficiency (%)',
    'Mean Fluorescence Intensity (MFI)', 'Positive Events (%)', 'Forward Scatter (FSC)',
    'Side Scatter (SSC)', 'Median (a.u.)', 'Geometric Mean (a.u.)',
    'Elastic Modulus (GPa)', 'Viscosity (cP)', 'Surface Tension (mN/m)', 'Contact Angle (°)',
    'Zeta Potential (mV)', 'Particle Size (nm)', 'Polydispersity Index (PDI)',
    'Crystallinity (%)', 'Porosity (%)', 'Surface Area (m²/g)', 'Pore Volume (cm³/g)',
    'Conductivity (S/cm)', 'Resistance (Ω)', 'Capacitance (F)', 'Impedance (Ω)',
    'Power (W)', 'Energy (J)', 'Frequency (Hz)', 'Amplitude (V)',
    'Phase Shift (°)', 'Dielectric Constant', 'Loss Tangent', 'Permittivity',
    'Transmittance (%)', 'Reflectance (%)', 'Scattering Coefficient', 'Extinction Coefficient',
    'Refractive Index (n)', 'Birefringence (Δn)', 'Dichroism (ΔA)', 'Circular Dichroism (mdeg)',
    'NMR Chemical Shift (ppm)', 'Coupling Constant (J, Hz)', 'Relaxation Time (T1, s)',
    'Diffusion Coefficient (D, m²/s)', 'Mobility (μ, cm²/V·s)', 'Drift Velocity (cm/s)',
    'Partition Coefficient (LogP)', 'Solubility (mg/L)', 'Dissolution Rate (%/min)',
    'Hardness (HV)', 'Tensile Strength (MPa)', 'Elongation (%)', 'Fracture Toughness (MPa·m½)',
    'Thermal Conductivity (W/m·K)', 'Heat Capacity (J/g·K)', 'Glass Transition (Tg, °C)',
    'Degradation Rate (%/day)', 'Swelling Ratio (%)', 'Water Uptake (%)',
    'Contact Time (min)', 'Residence Time (min)', 'Space Velocity (h⁻¹)', 'Conversion Rate (mol/s)',
    'Selectivity Index', 'Separation Factor (α)', 'Enrichment Factor', 'Detection Limit (ng/mL)',
    'Quantification Limit (ng/mL)', 'Precision (%RSD)', 'Accuracy (%)', 'Linearity (R²)',
    'Robustness Score', 'Reproducibility (%)', 'Intermediate Precision (%)', 'Specificity Score'
]


COMPARATIVE_LABELS = [
    ("Drug A", "Cmpd X"),
    ("Vehicle", "Test Article"),
    ("Control", "Treatment"),
    ("Placebo", "Sham"),
    ("Baseline", "Pre-treatment"),
    ("Endpoint", "Primary Outcome"),
    ("Subject", "Participant"),
    ("Dose", "Concentration"),
    ("Sample", "Specimen"),
    ("Protocol", "Method"),
    ("Adverse Event", "Side Effect"),
    ("Efficacy", "Effectiveness"),
    ("Randomized", "Allocated"),
    ("Blinded", "Masked"),
    ("Washout", "Clearance Period"),
    ("Cohort", "Group"),
    ("Intervention", "Exposure"),
    ("Response", "Outcome"),
    ("Biomarker", "Indicator"),
    ("Screening", "Enrollment"),
    ("Follow-up", "Monitoring"),
    ("Discontinuation", "Withdrawal"),
    ("Concomitant", "Concurrent"),
    ("Standard of Care", "Current Treatment"),
    ("Investigational", "Experimental"),
    ("Active Comparator", "Reference Treatment"),
    ("Primary Analysis", "Main Analysis"),
    ("Per Protocol", "Compliant Population"),
    ("Intent-to-Treat", "All Randomized"),
    ("Safety Population", "Exposed Subjects")
]


BUSINESS_Y_LABELS = [
    'Category', 'Product', 'SKU', 'Region', 'Country', 'State', 'City', 'Store',
    'Department', 'Salesperson', 'Marketing Channel', 'Quarter', 'Month', 'Week',
    'Year', 'Fiscal Quarter', 'Campaign', 'Channel', 'Platform', 'Device', 'Segment',
    'Customer Cohort', 'Cohort Start Date', 'Lead Source', 'Acquisition Channel',
    'Promotion', 'Price Band', 'Tier', 'Contract Length (months)', 'Project',
    'Funnel Stage', 'Audience', 'Persona', 'Order Date', 'Shipping Method', 'Distribution Center',
    'Territory', 'Season', 'Fiscal Week', 'Business Unit','Sales ($M)', 'Revenue (USD)', 'Revenue ($)', 'Units Sold (k)', 'Units Sold',
    'Market Share (%)', 'Customer Count', 'Active Users', 'Monthly Active Users (MAU)',
    'Daily Active Users (DAU)', 'Click-Through Rate (%)', 'Conversion Rate (%)',
    'Cost per Acquisition ($)', 'Cost per Click (CPC $)', 'Cost of Goods Sold (COGS $)',
    'Gross Profit ($)', 'Net Income ($)', 'EBITDA ($)', 'Operating Expenses ($)',
    'Profit Margin (%)', 'Gross Margin (%)', 'Average Order Value ($)',
    'Customer Lifetime Value (LTV $)', 'Churn Rate (%)', 'Retention Rate (%)',
    'Return on Investment (ROI %)', 'Return on Ad Spend (ROAS)', 'Inventory (units)',
    'Stock Level (%)', 'Fulfillment Time (days)', 'Lead Time (days)',
    'Bounce Rate (%)', 'Engagement Rate (%)', 'Page Views', 'Impressions', 'Clicks',
    'Session Duration (min)', 'Revenue per User ($)', 'CAC ($)', 'Pipeline Value ($)',
    'Forecast Accuracy (%)', 'Net Promoter Score (NPS)', 'Customer Satisfaction (CSAT)',
    'Employee Headcount', 'Revenue per Employee ($)', 'Avg. Resolution Time (hrs)'
]

SCIENTIFIC_X_LABELS = [
    'Group', 'Treatment', 'Condition', 'Dose (mg/kg)', 'Dose (μg/mL)', 'Time Point (h)',
    'Time Point (min)', 'Concentration (μM)', 'Sample ID', 'Replicate', 'Batch',
    'Genotype', 'Phenotype', 'Cell Line', 'Species', 'Subject ID', 'Age (years)',
    'Sex', 'Location', 'Site', 'Visit Number', 'Passage Number', 'Culture Duration (hrs)',
    'Temperature (°C)', 'pH', 'Stimulus', 'Protocol', 'Experimental Phase', 'Arm',
    'Trial', 'Environmental Factor', 'Exposure Time (s)', 'Wavelength (nm)', 'Frequency (Hz)',
    'Observation Date', 'Collection Time', 'Severity', 'Treatment Arm', 'Sequence ID',
        'Response (a.u.)', 'Activity (%)', 'Time (s)', 'Time (min)', 'Time (h)',
    'Concentration (μM)', 'Concentration (mg/L)', 'Molarity (mM)', 'Fold Change',
    'Optical Density (OD)', 'OD600', 'Viability (%)', 'Survival Fraction',
    'Expression Level (a.u.)', 'Signal (RFU)', 'Luminescence (RLU)', 'Absorbance (A.U.)',
    'Rate (units/s)', 'Enzyme Activity (U/mL)', 'Binding (%)', 'Binding Affinity (Kd, nM)',
    'IC50 (nM)', 'EC50 (μM)', 'Titer (ng/mL)', 'Copy Number', 'Reads per million (RPM)',
    'TPM', 'RPKM', 'Mutation Frequency (%)', 'Allele Frequency (%)', 'Colony Forming Units (CFU/mL)',
    'Cell Count (cells/mL)', 'Growth Rate (doublings/hr)', 'pH', 'Temperature (°C)',
    'Pressure (kPa)', 'Intensity (a.u.)', 'Wavelength (nm)', 'Density (g/mL)',
    'Mass (mg)', 'Volume (mL)', 'Torque (Nm)', 'Force (N)', 'Voltage (mV)',
    'Current (µA)', 'Signal-to-Noise Ratio', '% Inhibition', 'Normalized Score',
    'Fluorescence (AU)', 'FRET Ratio', 'Anisotropy (r)', 'Quantum Yield (%)',
    'Permeability (cm/s)', 'Clearance (mL/min)', 'Half-life (t1/2, h)', 'AUC (ng·h/mL)',
    'Cmax (ng/mL)', 'Bioavailability (%)', 'Protein Binding (%)', 'Recovery (%)',
    'Purity (%)', 'Yield (%)', 'Conversion (%)', 'Selectivity (%)',
    'Retention Time (min)', 'Peak Area (mAU·s)', 'Resolution (Rs)', 'Theoretical Plates (N)',
    'Ct Value', 'ΔΔCt', 'Melting Temperature (Tm, °C)', 'Amplification Efficiency (%)',
    'Mean Fluorescence Intensity (MFI)', 'Positive Events (%)', 'Forward Scatter (FSC)',
    'Side Scatter (SSC)', 'Median (a.u.)', 'Geometric Mean (a.u.)',
    'Elastic Modulus (GPa)', 'Viscosity (cP)', 'Surface Tension (mN/m)', 'Contact Angle (°)',
    'Zeta Potential (mV)', 'Particle Size (nm)', 'Polydispersity Index (PDI)',
    'Crystallinity (%)', 'Porosity (%)', 'Surface Area (m²/g)', 'Pore Volume (cm³/g)',
    'Conductivity (S/cm)', 'Resistance (Ω)', 'Capacitance (F)', 'Impedance (Ω)',
    'Power (W)', 'Energy (J)', 'Frequency (Hz)', 'Amplitude (V)',
    'Phase Shift (°)', 'Dielectric Constant', 'Loss Tangent', 'Permittivity',
    'Transmittance (%)', 'Reflectance (%)', 'Scattering Coefficient', 'Extinction Coefficient',
    'Refractive Index (n)', 'Birefringence (Δn)', 'Dichroism (ΔA)', 'Circular Dichroism (mdeg)',
    'NMR Chemical Shift (ppm)', 'Coupling Constant (J, Hz)', 'Relaxation Time (T1, s)',
    'Diffusion Coefficient (D, m²/s)', 'Mobility (μ, cm²/V·s)', 'Drift Velocity (cm/s)',
    'Partition Coefficient (LogP)', 'Solubility (mg/L)', 'Dissolution Rate (%/min)',
    'Hardness (HV)', 'Tensile Strength (MPa)', 'Elongation (%)', 'Fracture Toughness (MPa·m½)',
    'Thermal Conductivity (W/m·K)', 'Heat Capacity (J/g·K)', 'Glass Transition (Tg, °C)',
    'Degradation Rate (%/day)', 'Swelling Ratio (%)', 'Water Uptake (%)',
    'Contact Time (min)', 'Residence Time (min)', 'Space Velocity (h⁻¹)', 'Conversion Rate (mol/s)',
    'Selectivity Index', 'Separation Factor (α)', 'Enrichment Factor', 'Detection Limit (ng/mL)',
    'Quantification Limit (ng/mL)', 'Precision (%RSD)', 'Accuracy (%)', 'Linearity (R²)',
    'Robustness Score', 'Reproducibility (%)', 'Intermediate Precision (%)', 'Specificity Score'
]

BUSINESS_X_LABELS = [
    'Category', 'Product', 'SKU', 'Region', 'Country', 'State', 'City', 'Store',
    'Department', 'Salesperson', 'Marketing Channel', 'Quarter', 'Month', 'Week',
    'Year', 'Fiscal Quarter', 'Campaign', 'Channel', 'Platform', 'Device', 'Segment',
    'Customer Cohort', 'Cohort Start Date', 'Lead Source', 'Acquisition Channel',
    'Promotion', 'Price Band', 'Tier', 'Contract Length (months)', 'Project',
    'Funnel Stage', 'Audience', 'Persona', 'Order Date', 'Shipping Method', 'Distribution Center',
    'Territory', 'Season', 'Fiscal Week', 'Business Unit','Sales ($M)', 'Revenue (USD)', 'Revenue ($)', 'Units Sold (k)', 'Units Sold',
    'Market Share (%)', 'Customer Count', 'Active Users', 'Monthly Active Users (MAU)',
    'Daily Active Users (DAU)', 'Click-Through Rate (%)', 'Conversion Rate (%)',
    'Cost per Acquisition ($)', 'Cost per Click (CPC $)', 'Cost of Goods Sold (COGS $)',
    'Gross Profit ($)', 'Net Income ($)', 'EBITDA ($)', 'Operating Expenses ($)',
    'Profit Margin (%)', 'Gross Margin (%)', 'Average Order Value ($)',
    'Customer Lifetime Value (LTV $)', 'Churn Rate (%)', 'Retention Rate (%)',
    'Return on Investment (ROI %)', 'Return on Ad Spend (ROAS)', 'Inventory (units)',
    'Stock Level (%)', 'Fulfillment Time (days)', 'Lead Time (days)',
    'Bounce Rate (%)', 'Engagement Rate (%)', 'Page Views', 'Impressions', 'Clicks',
    'Session Duration (min)', 'Revenue per User ($)', 'CAC ($)', 'Pipeline Value ($)',
    'Forecast Accuracy (%)', 'Net Promoter Score (NPS)', 'Customer Satisfaction (CSAT)',
    'Employee Headcount', 'Revenue per Employee ($)', 'Avg. Resolution Time (hrs)'
]

CHART_TITLES = [
    'Experimental Results', 'Dose–Response Curve', 'Time Series', 'Growth Curve',
    'Survival Analysis', 'Quarterly Performance', 'Monthly Revenue vs Target',
    'Comparative Analysis', 'Key Metrics Overview', 'Treatment Response',
    'A/B Test Results', 'Conversion Funnel', 'Cohort Retention', 'Customer Acquisition Funnel',
    'Campaign Performance', 'Sales by Region', 'Product Mix', 'Inventory Levels',
    'Forecast vs Actual', 'Variance Analysis', 'Feature Importance', 'KPI Dashboard',
    'Engagement Over Time', 'User Activity Heatmap', 'Correlation Matrix',
    'Distribution of Values', 'Boxplot by Group', 'Regression Fit', 'Residuals Plot',
    'Top Customers', 'Revenue Breakdown', 'Churn Analysis', 'Segmentation Overview',
    'Operational Metrics', 'Quality Control Chart', 'Benchmarking Report',
    'Resource Utilization', 'Production Yield', 'Experiment Summary', 'Statistical Summary',
    'Network Overview', 'Seasonal Trend', 'Risk Assessment', 'Sensor Readings Over Time',
    'Executive Summary Dashboard', 'Waterfall Chart', 'Pareto Analysis',
    'Clinical Trial Outcomes', 'Pharmacokinetic Profile', 'Biomarker Expression',
    'Drug Screening Results', 'Assay Validation', 'Method Comparison Study',
    'Stability Testing', 'Batch Release Data', 'Process Optimization',
    'Analytical Method Performance', 'Protein Purification', 'Cell Culture Growth',
    'Gene Expression Analysis', 'Pathway Enrichment', 'Metabolomics Profile',
    'Proteomics Heatmap', 'Mass Spectrometry Results', 'Chromatography Analysis',
    'Spectroscopy Data', 'Microscopy Quantification', 'Flow Cytometry Analysis',
    'Financial Performance Summary', 'Budget vs Actual', 'Cash Flow Analysis',
    'Profitability Analysis', 'Market Share Trends', 'Competitive Landscape',
    'Investment Portfolio Performance', 'Risk-Return Analysis', 'Credit Risk Assessment',
    'Liquidity Ratios', 'Debt-to-Equity Analysis', 'ROI Comparison',
    'Customer Lifetime Value', 'Marketing Attribution', 'Lead Generation Metrics',
    'Social Media Analytics', 'Email Campaign Performance', 'Website Traffic Analysis',
    'Search Engine Performance', 'Brand Awareness Study', 'Market Research Findings',
    'Survey Response Analysis', 'Net Promoter Score', 'Customer Satisfaction Trends',
    'Supply Chain Analytics', 'Manufacturing Efficiency', 'Quality Assurance Metrics',
    'Equipment Downtime Analysis', 'Energy Consumption Trends', 'Safety Incident Reports',
    'Environmental Impact Assessment', 'Sustainability Metrics', 'Waste Reduction Analysis',
    'Employee Performance Review', 'Training Effectiveness', 'Recruitment Analytics',
    'Patient Demographics', 'Treatment Efficacy', 'Adverse Event Analysis',
    'Hospital Readmission Rates', 'Diagnostic Accuracy', 'Healthcare Utilization',
    'Epidemic Surveillance', 'Vaccination Coverage', 'Public Health Indicators',
    'Machine Learning Model Performance', 'Algorithm Comparison', 'Cross-Validation Results',
    'Hyperparameter Optimization', 'Learning Curve Analysis', 'Confusion Matrix',
    'ROC Curve Analysis', 'Precision-Recall Curve', 'Feature Selection Results',
    'Clustering Analysis', 'Dimensionality Reduction', 'Anomaly Detection Results',
    'Time Series Forecasting', 'Trend Decomposition', 'Seasonal Adjustment',
    'Moving Average Analysis', 'Volatility Analysis', 'Price Performance Comparison',
    'Geographic Distribution', 'Demographic Breakdown', 'Population Study',
    'Environmental Monitoring', 'Climate Data Analysis', 'Weather Pattern Study',
    'Research Publication Trends', 'Citation Analysis', 'Collaboration Network',
    'Technology Adoption Curve', 'Innovation Pipeline', 'Product Development Timeline'
]

HISTOGRAM_Y_LABELS = [
    'Frequency', 'Count', 'Absolute Frequency', 'Relative Frequency',
    'Density', 'Probability Density', 'Normalized Frequency', 'Percentage (%)',
    'Relative Frequency (%)', 'Proportion', 'Cumulative Frequency',
    'Cumulative Percentage (%)', 'Cumulative Proportion', 'Relative Cumulative Frequency',
    'Number of Observations', 'Number of Cases', 'Sample Count',
    'Frequency Distribution', 'Probability', 'Likelihood',
    'Bin Count', 'Occurrence', 'Distribution', 'Normalized Count',
    'Weighted Frequency', 'Expected Frequency', 'Observed Frequency',
    'Events per Bin', 'Cases per Interval', 'Observations per Class',
    'Population Frequency', 'Sample Frequency', 'Group Frequency',
    'Frequency per Unit', 'Rate per Interval', 'Incidence',
    'Frequency per 1000', 'Frequency per Million', 'Parts per Million (ppm)',
    'Normalized Density', 'Kernel Density', 'Empirical Density',
    'Log Frequency', 'Log Count', 'Square Root Frequency',
    'Standardized Frequency', 'Z-Score Frequency', 'Weighted Count',
    'Adjusted Frequency', 'Bootstrap Frequency', 'Resampled Count'
]


THEMES = {
    'default': {
        'facecolor': 'white',
        'grid_color': '#CCCCCC',
        'grid_style': '--',
        'grid_linewidth': 0.8,
        'font': 'DejaVu Sans',
        'font_size': 10,
        'palette': 'viridis',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#333333',
        'legend_frame': False
    },

    'excel': {
        'facecolor': '#F2F2F2',
        'grid_color': 'white',
        'grid_style': '-',
        'grid_linewidth': 1.5,
        'font': 'Calibri',
        'font_size': 11,
        'palette': ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47'],
        'line_width': 2.0,
        'marker_size': 7,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#1F4E79',
        'legend_frame': True,
        'legend_facecolor': 'white'
    },

    'ggplot': {
        'facecolor': '#EBEBEB',
        'grid_color': 'white',
        'grid_style': '-',
        'grid_linewidth': 1.0,
        'font': 'Arial',
        'font_size': 10,
        'palette': ['#F8766D', '#7CAE00', '#00BFC4', '#C77CFF'],
        'line_width': 1.25,
        'marker_size': 5,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': False},
        'axis_color': '#4D4D4D',
        'legend_frame': False
    },

    'prism': {
        'facecolor': 'white',
        'grid_color': '#E0E0E0',
        'grid_style': ':',
        'grid_linewidth': 1,
        'font': 'Arial',
        'font_size': 9,
        'palette': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3'],
        'line_width': 1.0,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'tick_direction': 'out',
        'legend_frame': False
    },
        'Rainbow': {
        'facecolor': 'white',
        'grid_color': '#E0E0E0',
        'grid_style': ':',
        'grid_linewidth': 1,
        'font': 'Arial',
        'font_size': 9,
        'palette': 'rainbow',
        'line_width': 1.0,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'tick_direction': 'out',
        'legend_frame': False
    },

    'powerpoint': {
        'facecolor': 'white',
        'grid_color': '#D9D9D9',
        'grid_style': '-',
        'font': 'Calibri',
        'font_size': 12,
        'palette': ['#5A9BD5', '#ED7D31', '#70AD47', '#FFC000', '#4472C4', '#9E480E'],
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'add_shadows': True,
        'line_width': 2.0,
        'marker_size': 8,
        'legend_frame': True
    },

    'minimal': {
        'facecolor': 'white',
        'grid_color': None,
        'grid_style': 'none',
        'font': 'Helvetica',
        'font_size': 10,
        'palette': ['#222222'],
        'line_width': 1.2,
        'marker_size': 5,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': True},
        'axis_color': '#222222',
        'legend_frame': False
    },
    
    'pastel': {
        'facecolor': 'white',
        'grid_color': '#F4F4F4',
        'grid_style': '--',
        'font': 'Calibri',
        'font_size': 11,
        'palette': ['#A3C4DC', '#F6C4C9', '#FFD8A9', '#C7EFCF', '#D5C6E0'],
        'line_width': 1.0,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'colorblind_friendly': {
        'facecolor': 'white',
        'grid_color': '#EDEDED',
        'grid_style': '--',
        'font': 'Arial',
        'font_size': 10,
        'palette': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],  # Okabe-Ito / Tol-like
        'line_width': 1.6,
        'marker_size': 7,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'retro': {
        'facecolor': '#FFF8E7',
        'grid_color': '#E6D6B5',
        'grid_style': ':',
        'font': 'Georgia',
        'font_size': 11,
        'palette': ['#E4572E', '#17BEBB', '#FFC914', '#2E4057'],
        'line_width': 1.8,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#3A3A3A',
        'legend_frame': True,
        'legend_facecolor': '#FFF8E7'
    },

    'seaborn_like': {
        'facecolor': '#F5F5F5',
        'grid_color': '#EDEDED',
        'grid_style': '-',
        'font': 'Helvetica',
        'font_size': 10,
        'palette': 'tab10',
        'line_width': 1.4,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#333333',
        'legend_frame': False
    },

    'presentation_bold': {
        'facecolor': 'white',
        'grid_color': '#DDDDDD',
        'grid_style': '-',
        'font': 'Montserrat',
        'font_size': 14,
        'palette': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
        'line_width': 3.0,
        'marker_size': 10,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': True
    },

    'clinical': {
        'facecolor': 'white',
        'grid_color': '#F0F0F0',
        'grid_style': '--',
        'font': 'Arial',
        'font_size': 9,
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'line_width': 1.2,
        'marker_size': 5,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'errorbar_capsize': 3,
        'legend_frame': False
    },

    'heatmap': {
        'facecolor': 'white',
        'grid_color': None,
        'grid_style': 'none',
        'font': 'DejaVu Sans',
        'font_size': 9,
        'palette': 'inferno',
        'line_width': 0.6,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': False},
        'colorbar': {'orientation': 'vertical', 'shrink': 0.8},
        'legend_frame': False
    },

    'corporate': {
        'facecolor': '#FAFAFA',
        'grid_color': '#E8E8E8',
        'grid_style': '-',
        'font': 'Roboto',
        'font_size': 11,
        'palette': ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9'],
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': True,
        'legend_facecolor': '#FFFFFF'
    },

    'map_atlas': {
        'facecolor': '#F8F8F8',
        'grid_color': '#EDEDED',
        'grid_style': '--',
        'font': 'Liberation Sans',
        'font_size': 9,
        'palette': ['#2b8cbe', '#7bccc4', '#edf8b1'],
        'line_width': 0.8,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': False},
        'legend_frame': False
    },

    'accessible_compact': {
        'facecolor': 'white',
        'grid_color': '#EAEAEA',
        'grid_style': '--',
        'font': 'Arial',
        'font_size': 12,
        'palette': ['#000000', '#1B9E77', '#D95F02', '#7570B3', '#E7298A'],
        'line_width': 2.0,
        'marker_size': 8,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': "#00000032",
        'legend_frame': True,
        'contrast_enhanced': True
    },
    'high_contrast_qualitative': {
        'facecolor': 'white',
        'grid_color': '#E0E0E0',
        'grid_style': '-',
        'font': 'Arial',
        'font_size': 11,
        'palette': 'Set1',
        'line_width': 1.8,
        'marker_size': 7,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

        'pastel_tones1': {
        'facecolor': '#FEFEFE',
        'grid_color': '#F0F0F0',
        'grid_style': '--',
        'font': 'Calibri',
        'font_size': 11,
        'palette': 'Pastel1',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'pastel_tones2': {
        'facecolor': '#FEFEFE',
        'grid_color': '#F0F0F0',
        'grid_style': '--',
        'font': 'Calibri',
        'font_size': 11,
        'palette': 'Pastel2',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'scientific_diverging': {
        'facecolor': 'white',
        'grid_color': '#DDDDDD',
        'grid_style': ':',
        'font': 'Helvetica',
        'font_size': 10,
        'palette': 'coolwarm',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },
}

PUBLICATION_THEMES = {
    'nature': {
        'font': 'Arial',
        'font_size': 7,
        'title_size': 8,
        'label_size': 7,
        'line_width': 0.5,
        'marker_size': 3,
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        'grid_style': 'none',
        'spine_width': 0.5,
        'figure_size': (3.3, 2.4),  # inches (single column)
        'dpi': 300,
        'legend_frame': False
    },

    'science': {
        'font': 'Helvetica',
        'font_size': 6,
        'title_size': 7,
        'label_size': 6,
        'line_width': 0.75,
        'marker_size': 3,
        'palette': ['#E31A1C', '#1F78B4', '#33A02C'],
        'grid_style': 'none',
        'spine_width': 0.75,
        'figure_size': (3.5, 2.5),
        'dpi': 300,
        'legend_frame': False
    },

    'cell': {
        'font': 'Helvetica',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.6,
        'marker_size': 3,
        'palette': ['#0072B2', '#D55E00', '#009E73'],
        'grid_style': 'none',
        'spine_width': 0.6,
        'figure_size': (6.5, 4.0),
        'dpi': 300,
        'legend_frame': True
    },

    'lancet': {
        'font': 'Times New Roman',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.8,
        'marker_size': 3,
        'palette': ['#000000', '#666666'],
        'grid_style': 'none',
        'spine_width': 0.8,
        'figure_size': (7.0, 4.5),
        'dpi': 300,
        'legend_frame': False
    },

    'pnas': {
        'font': 'Times New Roman',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.7,
        'marker_size': 3,
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'grid_style': 'none',
        'spine_width': 0.7,
        'figure_size': (5.0, 3.5),
        'dpi': 300,
        'legend_frame': False
    },

    'bmj': {
        'font': 'Arial',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.8,
        'marker_size': 3,
        'palette': ['#2C7FB8', '#7FCDBB', '#EDF8B1'],
        'grid_style': 'none',
        'spine_width': 0.8,
        'figure_size': (6.0, 4.0),
        'dpi': 300,
        'legend_frame': False
    },

    'nanotech_short': {
        'font': 'Helvetica',
        'font_size': 7,
        'title_size': 8,
        'label_size': 7,
        'line_width': 0.6,
        'marker_size': 2.5,
        'palette': ['#4E79A7', '#F28E2B', '#E15759'],
        'grid_style': 'none',
        'spine_width': 0.5,
        'figure_size': (3.2, 2.4),
        'dpi': 600,
        'legend_frame': False
    },

    'open_access_poster': {
        'font': 'Arial',
        'font_size': 10,
        'title_size': 12,
        'label_size': 10,
        'line_width': 1.0,
        'marker_size': 4,
        'palette': ['#1b9e77', '#d95f02', '#7570b3'],
        'grid_style': '--',
        'spine_width': 0.7,
        'figure_size': (8.0, 6.0),
        'dpi': 300,
        'legend_frame': True
    },

    'print_high_contrast': {
        'font': 'Times New Roman',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 1.0,
        'marker_size': 3,
        'palette': ['#000000', '#666666', '#AAAAAA'],
        'grid_style': 'none',
        'spine_width': 1.0,
        'figure_size': (6.0, 4.0),
        'dpi': 600,
        'legend_frame': False
    }

}

# Adicionado com base na análise para Diversidade de Tipografia
FONT_FAMILIES = {
    'sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'serif': ['Times New Roman', 'Georgia', 'DejaVu Serif'],
    'monospace': ['Courier New', 'DejaVu Sans Mono'], # Para rótulos de dados
}

# ===================================================================================
# == ENHANCED HEATMAP LABEL POOLS FOR DIVERSITY ==
# ===================================================================================

HEATMAP_XLABELS_SCIENTIFIC = [
    "Sample ID", "Replicate Number", "Time Point (h)", "Time Point (min)",
    "Treatment Dose (μg/mL)", "Dose Level (nM)", "Concentration (μM)",
    "Patient Cohort", "Experimental Condition", "Cell Line", "Tissue Type",
    "Gene Symbol", "Protein ID", "Metabolite", "Compound ID",
    "Observation Day", "Culture Passage", "Treatment Duration (h)",
    "Temperature (°C)", "pH Level", "Stimulus Intensity",
    "Wavelength (nm)", "Frequency (Hz)", "Channel Number",
    # additions
    "Experiment ID", "Plate Number", "Well ID", "Barcode", "Batch ID",
    "Run ID", "Instrument ID", "Operator", "Library Prep Kit",
    "Sequencing Depth (reads)", "Read Length (bp)", "Coverage (%)",
    "Library Batch", "Flowcell ID", "Lane Number", "Probe ID",
    "Antibody Clone", "Staining Protocol", "Microscope Objective",
    "Magnification (x)", "Pixel Size (μm)", "ROI ID", "Electrode ID",
    "Stimulus Frequency (Hz)", "Inoculation Route", "Dosage (mg/kg)",
    "Subject ID", "Age (years)", "Sex", "BMI (kg/m²)", "Ethnicity",
    "Comorbidity Flag", "Clinical Site", "Trial Arm", "Visit Number",
    "Post-dose Time (min)", "Pre-treatment Baseline", "Control Group",
    "Treatment Group", "Storage Condition", "Freeze-Thaw Cycles",
    "Extraction Method", "Assay Kit Lot", "Instrument Channel",
    "Sensor ID", "Acquisition Mode", "Imaging Modality"
]

HEATMAP_YLABELS_SCIENTIFIC = [
    "Gene Symbol", "Protein Target", "Transcript ID", "Pathway",
    "Cellular Component", "Biological Process", "Metabolite",
    "Biomarker", "Clinical Parameter", "Measurement Type",
    "Assay Replicate", "Technical Replicate", "Sample Group",
    "Treatment Arm", "Disease Stage", "Severity Score",
    "Feature Vector", "Principal Component", "Latent Factor",
    "Response Variable", "Outcome Measure",
    # additions
    "Gene Ontology Term", "Enzyme", "Reaction ID", "Metabolic Pathway",
    "SNP ID", "CpG Site", "Cell Type", "Subcellular Location",
    "Isoform", "Transcript Variant", "Protein Domain", "Protein Motif",
    "Binding Site", "Epitope", "Post-translational Mod", "Phosphosite",
    "Interaction Partner", "Protein Complex", "Phenotype",
    "Survival Time (days)", "Hazard Ratio", "Adverse Event", "IC50 (nM)",
    "EC50 (μM)", "LOD (a.u.)", "LOQ", "Signal-to-Background",
    "Normalized Count (TPM)", "RPKM", "CPM", "Delta Ct", "Ct Value",
    "Z-score", "t-statistic", "Adjusted p-value (FDR)", "Enrichment Term",
    "Cluster Label", "Module ID", "Annotation", "Taxon ID", "Strain"
]

HEATMAP_XLABELS_BUSINESS = [
    "Product SKU", "Sales Region", "Customer Segment", "Marketing Channel",
    "Campaign ID", "Fiscal Quarter", "Business Week", "Month",
    "Store Location", "Distribution Center", "Sales Territory",
    "Customer Cohort", "Account Type", "Service Tier",
    "Device Type", "Platform", "Acquisition Source",
    # additions
    "Order ID", "Invoice Period", "Sales Rep", "Promotion Type",
    "Price Tier", "List Price ($)", "Discount (%)", "Bundle ID",
    "Customer Lifetime Stage", "Subscription Plan", "Billing Cycle",
    "Payment Method", "Fulfillment Channel", "Return Reason",
    "Lead Source", "Landing Page", "Ad Group", "Keyword",
    "Traffic Channel", "Session Source", "Session Medium", "Browser",
    "Operating System", "App Version", "Feature Flag", "Product Line",
    "Product Family", "Manufacturing Batch", "Supplier ID", "PO Number",
    "Shipping Method", "Delivery Window", "Stock Keeping Unit",
    "Warehouse Zone", "Shelf Location", "CSR Team", "Contract Type"
]

HEATMAP_YLABELS_BUSINESS = [
    "Key Performance Indicator", "Revenue Stream", "Cost Center",
    "Product Category", "Service Line", "Business Unit",
    "Performance Metric", "Engagement Metric", "Conversion Stage",
    "Customer Attribute", "Behavioral Segment", "Risk Category",
    "Quality Metric", "Operational KPI", "Financial Ratio",
    # additions
    "Customer Lifetime Value", "Average Order Value", "Churn Rate",
    "Retention Cohort", "Funnel Stage", "Lead Quality", "Sales Velocity",
    "Support Ticket Type", "SLA Breach", "Inventory Level",
    "Days of Inventory", "Stockout Indicator", "Supplier Performance",
    "On-time Delivery Rate", "Return Rate", "Warranty Claims",
    "Defect Rate", "Cycle Time", "Throughput", "Downtime Minutes",
    "Utilization", "Headcount", "Cost per Employee", "Training Hours",
    "Compliance Flag", "Risk Score", "Fraud Indicator", "Net Revenue",
    "Gross Margin", "Operating Expense Category"
]

COLORBAR_TITLES_SCIENTIFIC = [
    "Expression Level (log₂)",
    "Normalized Intensity (z-score)",
    "Fold Change (log₂ FC)",
    "Correlation Coefficient (r)",
    "p-value (-log₁₀)",
    "Percent Change (%)",
    "Relative Abundance",
    "Signal-to-Noise Ratio",
    "Effect Size (Cohen's d)",
    "Normalized Response",
    "Activity Score (0-1)",
    "Enrichment Score",
    "Similarity Index",
    "Distance Metric",
    "Probability Density",
    "Rate Constant (s⁻¹)",
    "Concentration (μM)",
    "Intensity (a.u.)",
    "Fluorescence (RFU)",
    "Absorbance (OD)",
    # additions
    "Coverage (%)", "Read Depth (reads)", "Methylation (%)", "Beta Value",
    "Counts per Million (CPM)", "TPM", "RPKM", "Delta Ct", "Ct Value",
    "Log Odds", "Binding Affinity (K_D, nM)", "Enrichment (NES)",
    "Adjusted p-value (q)", "Probability (0-1)", "Likelihood Ratio",
    "Velocity (mm/s)", "Frequency (Hz)", "Optical Density (AU)",
    "Normalized Counts", "Signal Intensity (counts)"
]

COLORBAR_TITLES_BUSINESS = [
    "Revenue ($M)",
    "Growth Rate (%)",
    "Market Share (%)",
    "Customer Satisfaction (1-10)",
    "Net Promoter Score",
    "Conversion Rate (%)",
    "Engagement Score",
    "Performance Index",
    "Efficiency Ratio",
    "Return on Investment (%)",
    "Cost per Acquisition ($)",
    "Lifetime Value ($K)",
    "Churn Probability (%)",
    "Inventory Turnover",
    "Fill Rate (%)",
    "Response Time (min)",
    "Utilization Rate (%)",
    "Quality Score (0-100)",
    # additions
    "Average Order Value ($)", "ARPU ($)", "MRR ($)",
    "Gross Margin (%)", "Operating Margin (%)", "EBITDA ($M)",
    "Days Sales Outstanding (DSO)", "Customer Retention Rate (%)",
    "CAC Payback (months)", "Cart Abandonment Rate (%)",
    "Return on Ad Spend (ROAS)", "Cost of Goods Sold ($)",
    "Payroll Expense ($K)", "Revenue per Employee ($K)",
    "Inventory Days", "Defect Rate (%)", "On-time Delivery (%)",
    "First Response Time (min)", "SLA Compliance (%)", "Fraud Rate (%)"
]

HEATMAP_CHART_TITLES = [
    "Correlation Matrix",
    "Gene Expression Heatmap",
    "Sample Clustering Analysis",
    "Time-Course Response Pattern",
    "Treatment Effect Profile",
    "Cross-Feature Association Map",
    "Hierarchical Clustering Result",
    "Distance Matrix Visualization",
    "Temporal Activity Pattern",
    "Multi-Variable Comparison",
    "Performance Scorecard",
    "Regional Sales Performance",
    "Customer Behavior Matrix",
    "Product Performance Comparison",
    "Cohort Analysis Dashboard",
    "A/B Test Results Matrix",
    "Channel Attribution Heatmap",
    "Seasonal Trend Analysis",
    "Risk Assessment Matrix",
    "Quality Control Dashboard",
    # additions
    "Supply Chain Heatmap", "Inventory Level Over Time",
    "Sales Funnel Conversion Matrix", "Customer Lifetime Value Map",
    "Employee Performance Matrix", "Service Latency Heatmap",
    "Model Confusion Matrix", "Hyperparameter Sweep Results",
    "Feature Importance Matrix", "Anomaly Detection Heatmap",
    "Genotype-Phenotype Association Map", "Protein-Protein Interaction Map",
    "Metabolomics Profile Heatmap", "Pharmacokinetic Heatmap",
    "Electrophysiology Activity Map", "Brain Activation Map",
    "Operational Risk Heatmap", "Fraud Detection Score Matrix",
    "Retention Cohort Heatmap", "Market Penetration Map",
    "Channel vs Product Performance"
]

HEATMAP_ANNOTATION_FORMATS = [
    "{:.2f}",  # Two decimal places
    "{:.1f}",  # One decimal place
    "{:.3f}",  # Three decimal places
    "{:.0f}",  # Integer
    "{:.1%}",  # Percentage with one decimal
    "{:.0%}",  # Percentage integer
    "{:.2e}",  # Scientific notation
]
