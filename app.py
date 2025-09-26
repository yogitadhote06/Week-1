{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "20b03c75-d070-4f3b-a282-210a31618443",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-09-14 16:52:04.281 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:04.283 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.324 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-09-14 16:52:05.326 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.327 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.329 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.330 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.333 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.334 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.335 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.338 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.341 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.343 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.344 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.346 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-09-14 16:52:05.348 Session state does not function when running a script without `streamlit run`\n"
     ]
    },
    {
     "ename": "StreamlitValueAboveMaxError",
     "evalue": "The `value` 10 is greater than the `max_value` 8.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mStreamlitValueAboveMaxError\u001b[0m               Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 44\u001b[0m\n\u001b[0;32m     33\u001b[0m     data \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m     34\u001b[0m         \u001b[38;5;66;03m#'Make': cat_dict['Make'].tolist().index(make),\u001b[39;00m\n\u001b[0;32m     35\u001b[0m         \u001b[38;5;66;03m#'Model': cat_dict['Model'].tolist().index(model),\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     39\u001b[0m         \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mEngine Size\u001b[39m\u001b[38;5;124m'\u001b[39m:enginesize\n\u001b[0;32m     40\u001b[0m     }\n\u001b[0;32m     42\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m pd\u001b[38;5;241m.\u001b[39mDataFrame([data])\n\u001b[1;32m---> 44\u001b[0m input_df \u001b[38;5;241m=\u001b[39m get_user_input()\n\u001b[0;32m     46\u001b[0m \u001b[38;5;66;03m#Prediction Button\u001b[39;00m\n\u001b[0;32m     47\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m st\u001b[38;5;241m.\u001b[39mbutton(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPredict CO2 Emission Rate\u001b[39m\u001b[38;5;124m\"\u001b[39m):\n\u001b[0;32m     48\u001b[0m     \u001b[38;5;66;03m#Optional: scale the input if needed\u001b[39;00m\n\u001b[0;32m     49\u001b[0m     \u001b[38;5;66;03m# scaled_input = scaler.transform(input_df)\u001b[39;00m\n",
      "Cell \u001b[1;32mIn[1], line 29\u001b[0m, in \u001b[0;36mget_user_input\u001b[1;34m()\u001b[0m\n\u001b[0;32m     20\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21mget_user_input\u001b[39m():\n\u001b[0;32m     21\u001b[0m     \u001b[38;5;66;03m#'Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders',\u001b[39;00m\n\u001b[0;32m     22\u001b[0m       \u001b[38;5;66;03m# 'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100 km)',\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     27\u001b[0m     \u001b[38;5;66;03m#vehicleclass = st.sidebar.selectbox(\"Vehicle Class\", cat_dict['Vehicle Class'])\u001b[39;00m\n\u001b[0;32m     28\u001b[0m     \u001b[38;5;66;03m#fueltype = st.sidebar.selectbox(\"Fueltype\", cat_dict['Fueltype'])\u001b[39;00m\n\u001b[1;32m---> 29\u001b[0m     fuelconsumptionHwy \u001b[38;5;241m=\u001b[39m st\u001b[38;5;241m.\u001b[39msidebar\u001b[38;5;241m.\u001b[39mnumber_input(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFuel Consumption Hwy (L/100 km)\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m5\u001b[39m, \u001b[38;5;241m8\u001b[39m, \u001b[38;5;241m10\u001b[39m)\n\u001b[0;32m     30\u001b[0m     enginesize \u001b[38;5;241m=\u001b[39m st\u001b[38;5;241m.\u001b[39msidebar\u001b[38;5;241m.\u001b[39mnumber_input(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mEngine Size\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m2\u001b[39m,\u001b[38;5;241m3\u001b[39m,\u001b[38;5;241m4\u001b[39m)\n\u001b[0;32m     33\u001b[0m     data \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m     34\u001b[0m         \u001b[38;5;66;03m#'Make': cat_dict['Make'].tolist().index(make),\u001b[39;00m\n\u001b[0;32m     35\u001b[0m         \u001b[38;5;66;03m#'Model': cat_dict['Model'].tolist().index(model),\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     39\u001b[0m         \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mEngine Size\u001b[39m\u001b[38;5;124m'\u001b[39m:enginesize\n\u001b[0;32m     40\u001b[0m     }\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\streamlit\\runtime\\metrics_util.py:443\u001b[0m, in \u001b[0;36mgather_metrics.<locals>.wrapped_func\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    441\u001b[0m         _LOGGER\u001b[38;5;241m.\u001b[39mdebug(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFailed to collect command telemetry\u001b[39m\u001b[38;5;124m\"\u001b[39m, exc_info\u001b[38;5;241m=\u001b[39mex)\n\u001b[0;32m    442\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 443\u001b[0m     result \u001b[38;5;241m=\u001b[39m non_optional_func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    444\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m RerunException:\n\u001b[0;32m    445\u001b[0m     \u001b[38;5;66;03m# Duplicated from below, because static analysis tools get confused\u001b[39;00m\n\u001b[0;32m    446\u001b[0m     \u001b[38;5;66;03m# by deferring the rethrow.\u001b[39;00m\n\u001b[0;32m    447\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m tracking_activated \u001b[38;5;129;01mand\u001b[39;00m command_telemetry:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\widgets\\number_input.py:401\u001b[0m, in \u001b[0;36mNumberInputMixin.number_input\u001b[1;34m(self, label, min_value, max_value, value, step, format, key, help, on_change, args, kwargs, placeholder, disabled, label_visibility, icon, width)\u001b[0m\n\u001b[0;32m    237\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124mr\u001b[39m\u001b[38;5;124;03m\"\"\"Display a numeric input widget.\u001b[39;00m\n\u001b[0;32m    238\u001b[0m \n\u001b[0;32m    239\u001b[0m \u001b[38;5;124;03m.. note::\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    398\u001b[0m \n\u001b[0;32m    399\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    400\u001b[0m ctx \u001b[38;5;241m=\u001b[39m get_script_run_ctx()\n\u001b[1;32m--> 401\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_number_input(\n\u001b[0;32m    402\u001b[0m     label\u001b[38;5;241m=\u001b[39mlabel,\n\u001b[0;32m    403\u001b[0m     min_value\u001b[38;5;241m=\u001b[39mmin_value,\n\u001b[0;32m    404\u001b[0m     max_value\u001b[38;5;241m=\u001b[39mmax_value,\n\u001b[0;32m    405\u001b[0m     value\u001b[38;5;241m=\u001b[39mvalue,\n\u001b[0;32m    406\u001b[0m     step\u001b[38;5;241m=\u001b[39mstep,\n\u001b[0;32m    407\u001b[0m     \u001b[38;5;28mformat\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mformat\u001b[39m,\n\u001b[0;32m    408\u001b[0m     key\u001b[38;5;241m=\u001b[39mkey,\n\u001b[0;32m    409\u001b[0m     help\u001b[38;5;241m=\u001b[39mhelp,\n\u001b[0;32m    410\u001b[0m     on_change\u001b[38;5;241m=\u001b[39mon_change,\n\u001b[0;32m    411\u001b[0m     args\u001b[38;5;241m=\u001b[39margs,\n\u001b[0;32m    412\u001b[0m     kwargs\u001b[38;5;241m=\u001b[39mkwargs,\n\u001b[0;32m    413\u001b[0m     placeholder\u001b[38;5;241m=\u001b[39mplaceholder,\n\u001b[0;32m    414\u001b[0m     disabled\u001b[38;5;241m=\u001b[39mdisabled,\n\u001b[0;32m    415\u001b[0m     label_visibility\u001b[38;5;241m=\u001b[39mlabel_visibility,\n\u001b[0;32m    416\u001b[0m     icon\u001b[38;5;241m=\u001b[39micon,\n\u001b[0;32m    417\u001b[0m     width\u001b[38;5;241m=\u001b[39mwidth,\n\u001b[0;32m    418\u001b[0m     ctx\u001b[38;5;241m=\u001b[39mctx,\n\u001b[0;32m    419\u001b[0m )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\widgets\\number_input.py:545\u001b[0m, in \u001b[0;36mNumberInputMixin._number_input\u001b[1;34m(self, label, min_value, max_value, value, step, format, key, help, on_change, args, kwargs, placeholder, disabled, label_visibility, icon, width, ctx)\u001b[0m\n\u001b[0;32m    542\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m StreamlitValueBelowMinError(value\u001b[38;5;241m=\u001b[39mvalue, min_value\u001b[38;5;241m=\u001b[39mmin_value)\n\u001b[0;32m    544\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m max_value \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m value \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m max_value \u001b[38;5;241m<\u001b[39m value:\n\u001b[1;32m--> 545\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m StreamlitValueAboveMaxError(value\u001b[38;5;241m=\u001b[39mvalue, max_value\u001b[38;5;241m=\u001b[39mmax_value)\n\u001b[0;32m    547\u001b[0m \u001b[38;5;66;03m# Bounds checks. JSNumber produces human-readable exceptions that\u001b[39;00m\n\u001b[0;32m    548\u001b[0m \u001b[38;5;66;03m# we simply re-package as StreamlitAPIExceptions.\u001b[39;00m\n\u001b[0;32m    549\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n",
      "\u001b[1;31mStreamlitValueAboveMaxError\u001b[0m: The `value` 10 is greater than the `max_value` 8."
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "#Load required models and encoders\n",
    "model = joblib.load(\"C:/Users/HP/Downloads/my code/Python-ML/lr_model.pkl\")\n",
    "scaler = joblib.load(\"C:/Users/HP/Downloads/my code/Python-ML/scaler.pkl\")\n",
    "#cat_dict = joblib.load(\"cat_dict.pkl\")\n",
    "\n",
    "st.set_page_config(page_title=\"CO2 Emission Rate\")\n",
    "st.title(\"CO2 Emission Rate Prediction App\")\n",
    "st.markdown(\"Predict CO2 Emission Rate based on factors such as Fuel Type, etc\")\n",
    "\n",
    "# sidebar inputs\n",
    "st.sidebar.header(\"Input Parameters\")\n",
    "\n",
    "def get_user_input():\n",
    "    #'Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders',\n",
    "      # 'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100 km)',\n",
    "       #'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',\n",
    "       #'Fuel Consumption Comb (mpg)'\n",
    "    #make = st.sidebar.selectbox(\"Make\", cat_dict['Make'])\n",
    "    #model = st.sidebar.selectbox(\"Model\", cat_dict['Model'])\n",
    "    #vehicleclass = st.sidebar.selectbox(\"Vehicle Class\", cat_dict['Vehicle Class'])\n",
    "    #fueltype = st.sidebar.selectbox(\"Fueltype\", cat_dict['Fueltype'])\n",
    "    fuelconsumptionHwy = st.sidebar.number_input(\"Fuel Consumption Hwy (L/100 km)\", 5, 8, 10)\n",
    "    enginesize = st.sidebar.number_input(\"Engine Size\", 2,3,4)\n",
    "\n",
    "\n",
    "    data = {\n",
    "        #'Make': cat_dict['Make'].tolist().index(make),\n",
    "        #'Model': cat_dict['Model'].tolist().index(model),\n",
    "        #'Vehicle Class': cat_dict['Vehicle Class'].tolist().index(vehicleclass),\n",
    "        #'Fueltype': cat_dict['Fueltype'].tolist().index(fueltype),\n",
    "        'Fuel Consumption Hwy (L/100 km)':fuelconsumptionHwy,\n",
    "        'Engine Size':enginesize\n",
    "    }\n",
    "\n",
    "    return pd.DataFrame([data])\n",
    "\n",
    "input_df = get_user_input()\n",
    "\n",
    "#Prediction Button\n",
    "if st.button(\"Predict CO2 Emission Rate\"):\n",
    "    #Optional: scale the input if needed\n",
    "    # scaled_input = scaler.transform(input_df)\n",
    "    prediction = model.predict(input_df)\n",
    "\n",
    "    st.subheader(\"Prediction Result\")\n",
    "    st.write(f\"The estimated value is: .2f\")\n",
    "    \n",
    "                                                         "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5cac2f2e-b1bf-46cd-a87f-94562aa20200",
   "metadata": {},
   "outputs": [],
   "source": [
    "streamlit run "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
