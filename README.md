# Aadhaar Service Stress Risk Dashboard

## Project Objective
This project develops an interactive Streamlit dashboard to help judges and governance officials explore and understand Aadhaar service stress risks across Indian districts. By visualizing key metrics and trends, the dashboard facilitates data-driven decision-making for optimizing Aadhaar service delivery and resource allocation.

## How to Run the App Locally
1. Ensure you have Python installed (version 3.7 or higher).
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
4. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## What the Dashboard Demonstrates
- **Interactive Filtering**: Select specific states, districts, and dates to focus on particular areas of interest.
- **Key Metrics Display**: View critical indicators like Service Stress Risk Score, Biometric-to-Enrolment Ratio, and update pressures for children and the elderly.
- **Trend Analysis**: Observe how service stress risk evolves over time for selected districts.
- **Risk Ranking**: Identify the top 10 high-risk districts based on average stress scores.
- **Explainability**: Receive human-readable explanations of risk factors for better understanding.
- **Data Export**: Download ranked district stress data for further analysis.

## Why This Solution Matters for Governance
In a country with over 1.3 billion Aadhaar enrolments, efficient service delivery is crucial for citizen satisfaction and trust in government systems. This dashboard empowers decision-makers to:
- Proactively identify districts under stress before service breakdowns occur.
- Allocate resources (personnel, infrastructure) to high-risk areas.
- Monitor the impact of policy changes on service stress over time.
- Ensure equitable access to Aadhaar services across all demographics, particularly vulnerable groups like children and the elderly.

By providing clear, non-technical insights into complex data, this tool bridges the gap between data science and governance, enabling more responsive and effective public service management.