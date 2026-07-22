#  Hotel Booking Cancellation Risk Predictor

An end-to-end Machine Learning application designed to predict the probability of hotel booking cancellations in real-time. Built with **XGBoost**, **Scikit-Learn**, and **Streamlit**, this tool helps hotel operations and revenue managers reduce lost inventory costs and optimize dynamic overbooking strategies.

 **Live Interactive App:** [Hotel Cancellation Risk Predictor](https://hotel-booking-cancellation-prediction-vtpse79vcanprqx5zusjpv.streamlit.app/)

---

##  Executive Summary & Problem Statement

Hotel cancellations present a major challenge to the hospitality industry, leading to lost revenue, empty rooms, and inefficient operational planning. Last-minute cancellations leave inventory unallocated that could otherwise be resold.

**Objective:** Build a predictive ML model to assign a real-time **Cancellation Risk Score** to incoming bookings based on passenger history and reservation details, enabling proactive interventions (e.g., confirmation reminders, non-refundable deposit incentives, or controlled overbooking).

---

##  Key Features & Input Specifications

The model evaluates **10 key high-impact features** engineered from historical booking records:

| Feature Name | Description | Values / Range | Business Impact |
| :--- | :--- | :--- | :--- |
| **`hotel`** | Hotel Type | `City Hotel` (1), `Resort Hotel` (0) | City hotels typically experience higher cancellation rates. |
| **`lead_time`** | Days between booking & arrival | `0` to `700` days | Longer lead times correlate strongly with higher cancellation risk. |
| **`market_segment`** | Booking Channel | `Direct`, `Online TA`, `Offline TA/TO`, `Corporate`, `Groups` | Bookings via Online Travel Agencies (TA) have higher churn. |
| **`deposit_type`** | Deposit Policy | `No Deposit`, `Non Refundable`, `Refundable` | Non-refundable deposits drastically drop cancellation risk. |
| **`customer_type`** | Booking Category | `Transient`, `Contract`, `Transient-Party`, `Group` | Individual transient guests cancel more frequently. |
| **`previous_cancellations`** | Past cancellation count | `0` to `30` | Historical behavior is a primary predictor of future actions. |
| **`total_of_special_requests`** | Requests made (twin bed, floor) | `0` to `5` | Guests with specific requests show higher commitment. |
| **`required_car_parking_spaces`** | Parking requested | `No` (0), `Yes` (1) | Guests requesting parking almost always show up. |
| **`adr`** | Average Daily Rate ($) | `$0.0` to `$1000.0` | Higher nightly rates slightly increase cancellation likelihood. |
| **`is_repeated_guest`** | Loyalty Flag | `No` (0), `Yes` (1) | Repeat guests rarely cancel their reservations. |

---

##  Tech Stack & Workflow

* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Seaborn, Matplotlib
* **Machine Learning:** XGBoost Classifier, Scikit-Learn
* **Model Serialization:** Joblib
* **Frontend UI:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## Repository Structure

```text
├── app.py                         # Streamlit user interface & inference logic
├── hotel_cancellation_model.pkl   # Serialized trained XGBoost model
├── hotel_booking_Project.ipynb    # having complete model with training and prediction
├── requirements.txt               # App dependencies for cloud deployment
└── README.md                      # Comprehensive project documentation
