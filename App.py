import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

st.title("Hotel Booking Cancellation Risk Predictor")
st.write(
    "Predict room cancellation risk in real-time to optimize occupancy and"
    " revenue."
)
st.markdown("---")

@st.cache_resource
def load_model():
  data = joblib.load("hotel_cancellation_model.pkl")
  if isinstance(data, dict):
    return data.get("model"), data.get("scaler"), data.get("features")
  return data, None, None


try:
  model, scaler, feature_order = load_model()
  if model is None:
    raise ValueError("No model found inside hotel_cancellation_model.pkl")
except Exception as e:
  st.error(
      "❌ Could not load `hotel_cancellation_model.pkl`. Make sure the file is"
      " in the same folder and was saved by the (fixed) training notebook."
  )
  st.stop()
 
col1, col2 = st.columns(2)

with col1:
  st.subheader("📋 Stay & Booking Details")

  lead_time = st.number_input(
      "Lead Time (Days before arrival)", min_value=0, max_value=500, value=180
  )

  market_segment = st.selectbox(
      "Market Segment Type",
      options=["Online", "Offline", "Corporate", "Complementary", "Aviation"],
  )

  room_type = st.selectbox(
      "Room Type",
      options=[
          "Room_Type 1",
          "Room_Type 2",
          "Room_Type 3",
          "Room_Type 4",
          "Room_Type 5",
          "Room_Type 6",
          "Room_Type 7",
      ],
  )

  type_of_meal = st.selectbox(
      "Meal Plan",
      options=["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
  )

  weekend_nights = st.number_input(
      "Number of Weekend Nights", min_value=0, max_value=10, value=2
  )

  week_nights = st.number_input(
      "Number of Week Nights", min_value=0, max_value=20, value=4
  )

with col2:
  st.subheader("👤 Guest Profile & Behavior")

  avg_price = st.number_input(
      "Average Daily Rate ($ / Night)",
      min_value=0.0,
      max_value=1000.0,
      value=150.0,
  )

  total_guests = st.number_input(
      "Total Guests (Adults + Children)", min_value=1, max_value=10, value=2
  )

  special_requests = st.slider(
      "Number of Special Requests", min_value=0, max_value=5, value=0
  )

  car_parking = st.selectbox(
      "Car Parking Space Requested?", options=["No", "Yes"]
  )

  is_repeated = st.selectbox("Is Repeat Guest?", options=["No", "Yes"])

  p_not_c = st.number_input(
      "Previous Bookings Not Canceled", min_value=0, max_value=20, value=0
  )

 
meal_map = {
    "Meal Plan 1": 1,
    "Meal Plan 2": 2,
    "Meal Plan 3": 3,
    "Not Selected": 0,
}
room_map = {
    "Room_Type 1": 0,
    "Room_Type 2": 1,
    "Room_Type 3": 2,
    "Room_Type 4": 3,
    "Room_Type 5": 4,
    "Room_Type 6": 5,
    "Room_Type 7": 6,
}
market_map = {
    "Online": 0,
    "Aviation": 1,
    "Offline": 2,
    "Corporate": 3,
    "Complementary": 4,
}

meal_val = meal_map[type_of_meal]
room_val = room_map[room_type]
market_val = market_map[market_segment]
parking_val = 1 if car_parking == "Yes" else 0
repeat_val = 1 if is_repeated == "Yes" else 0

input_dict = {
    "number of weekend nights": weekend_nights,
    "number of week nights": week_nights,
    "type of meal": meal_val,
    "car parking space": parking_val,
    "room type": room_val,
    "lead time": lead_time,
    "market segment type": market_val,
    "repeated": repeat_val,
    "P-not-C": p_not_c,
    "average price": avg_price,
    "special requests": special_requests,
    "total_guests": total_guests,
}

input_df = pd.DataFrame([input_dict])

if feature_order:
  input_df = input_df[feature_order]

st.markdown("---")
if st.button("📊 Calculate Cancellation Risk", type="primary"):
  if scaler is not None:
    scaled_values = scaler.transform(input_df)
    model_input = pd.DataFrame(scaled_values, columns=input_df.columns)
  else:
    st.warning(
        "⚠️ No scaler found in the model file - predictions may be"
        " inaccurate. Please re-save the model using the fixed notebook."
    )
    model_input = input_df

  try:
    probability = float(model.predict_proba(model_input)[0][1] * 100)
  except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

  res_col1, res_col2 = st.columns([1, 2])
  is_high_risk = probability > 50.0

  with res_col1:
    st.metric(
        label="Cancellation Probability",
        value=f"{probability:.1f}%",
        delta="HIGH RISK" if is_high_risk else "LOW RISK",
        delta_color="inverse" if is_high_risk else "normal",
    )

  with res_col2:
    if is_high_risk:
      st.error(
          "⚠️ **High Cancellation Risk Detected!**\n\n"
          "**Actionable Insights:**\n"
          "• Send a booking re-confirmation notification.\n"
          "• Offer non-refundable prepaid discounts.\n"
          "• Consider strategic overbooking allowance for this date."
      )
    else:
      st.success(
          "✅ **Low Cancellation Risk!**\n\n"
          "**Actionable Insights:**\n"
          "• Guest exhibits strong stay intent.\n"
          "• Low likelihood of churn; proceed with standard check-in flow."
      )
