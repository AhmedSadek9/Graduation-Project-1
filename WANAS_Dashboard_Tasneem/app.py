
import json
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="WANAS Dashboard",
    page_icon="",
    layout="wide"
)


# ==========================================
# 2. LOAD PATIENT DATA
# ==========================================



DATA_FILE = Path(__file__).parent / "data" / "sample_data.json"
ALERTS_FILE = Path(__file__).parent / "data" / "alerts.json"

with open(DATA_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

patients = data["patients"]
def save_patient_data():
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {"patients": patients},
            file,
            indent=4
        )

# Select the first patient by default
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = patients[0]["id"]


# ==========================================
# SIDEBAR AND PATIENT SELECTION
# ==========================================

st.sidebar.title("WANAS")
st.sidebar.caption("Smart Alzheimer Care System")

st.sidebar.subheader("Select Patient")

patient_options = {
    p["name"]: p["id"] for p in patients
}

selected_patient_name = st.sidebar.selectbox(
    "Choose a patient",
    options=list(patient_options.keys()),
    index=list(patient_options.values()).index(
        st.session_state.selected_patient_id
    )
)

st.session_state.selected_patient_id = patient_options[selected_patient_name]

page = st.sidebar.radio(
    "Navigation",
    [
        "Patient Overview",
        "GPS & Safe Zones",
        "Medication Tracking",
        "Manage Medications",
        "Caregiver Alerts",
        "Event History",
        "Add New Patient",
        "Edit Patient Information"
    ]
)


# ==========================================
# LOAD THE SELECTED PATIENT
# ==========================================

patient = next(
    p for p in patients
    if p["id"] == st.session_state.selected_patient_id
)

gps = patient["gps"]
safe_zone = patient["safe_zone"]
medications = patient["medications"]

patient_id = patient["id"]


# ==========================================
# LOAD ALERT HISTORIES
# ==========================================

if "all_alert_histories" not in st.session_state:

    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as file:
                saved_alerts = json.load(file)

            if isinstance(saved_alerts, list):
                st.session_state.all_alert_histories = {
                    "WANAS-001": saved_alerts
                }
            else:
                st.session_state.all_alert_histories = saved_alerts

        except (json.JSONDecodeError, OSError):
            st.session_state.all_alert_histories = {}

    else:
        st.session_state.all_alert_histories = {}


if patient_id not in st.session_state.all_alert_histories:
    st.session_state.all_alert_histories[patient_id] = []


# ==========================================
# STORE SEPARATE GPS SETTINGS FOR EACH PATIENT
# ==========================================

if "patient_states" not in st.session_state:
    st.session_state.patient_states = {}

if patient_id not in st.session_state.patient_states:
    st.session_state.patient_states[patient_id] = {
        "last_safe_zone_state": None,
        "gps_connection_status": gps["status"],
        "last_gps_status": gps["status"],
        "patient_latitude": gps["latitude"],
        "patient_longitude": gps["longitude"]
    }


# Load the selected patient's saved settings
patient_state = st.session_state.patient_states[patient_id]

st.session_state.alert_history = (
    st.session_state.all_alert_histories[patient_id]
)

st.session_state.last_safe_zone_state = (
    patient_state["last_safe_zone_state"]
)

st.session_state.gps_connection_status = (
    patient_state["gps_connection_status"]
)

st.session_state.last_gps_status = (
    patient_state["last_gps_status"]
)

st.session_state.patient_latitude = (
    patient_state["patient_latitude"]
)

st.session_state.patient_longitude = (
    patient_state["patient_longitude"]
)


# Apply the selected patient's GPS settings
gps["status"] = st.session_state.gps_connection_status
gps["latitude"] = st.session_state.patient_latitude
gps["longitude"] = st.session_state.patient_longitude

# ==========================================
# 3. CALCULATE DISTANCE FROM SAFE ZONE
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance in meters between
    two GPS coordinates using the Haversine formula.
    """

    earth_radius = 6371000

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    difference_lat = lat2 - lat1
    difference_lon = lon2 - lon1

    a = (
        math.sin(difference_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(difference_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius * c


def save_alerts():
    # Save the current patient's alert history
    patient_id = st.session_state.selected_patient_id

    st.session_state.all_alert_histories[patient_id] = (
        st.session_state.alert_history
    )

    # Save all patients' alert histories together
    with open(ALERTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            st.session_state.all_alert_histories,
            file,
            indent=4
        )
def create_alert(alert_type, message, priority):

    patient_id = st.session_state.selected_patient_id

    alert_number = len(st.session_state.alert_history) + 1

    new_alert = {
        "id": f"{patient_id}-ALERT-{alert_number:03d}",
        "patient_id": patient_id,
        "patient_name": patient["name"],
        "type": alert_type,
        "message": message,
        "timestamp": datetime.now().astimezone().strftime(
            "%Y-%m-%d %I:%M:%S %p"
        ),
        "priority": priority,
        "status": "Unacknowledged"
    }

    # Add the newest alert at the beginning of the list
    st.session_state.alert_history.insert(0, new_alert)

    save_alerts()


def check_safe_zone_alert(is_inside, distance, safe_zone):

    if is_inside:
        st.session_state.last_safe_zone_state = "inside"

    else:
        # Create an alert only when the patient first leaves the safe zone
        if st.session_state.last_safe_zone_state != "outside":

            create_alert(
                alert_type="Safe Zone Violation",
                message=(
                    f"The patient is {distance:.1f} meters from home, "
                    f"which exceeds the safe-zone radius of "
                    f"{safe_zone['radius_m']} meters."
                ),
                priority="High"
            )

        st.session_state.last_safe_zone_state = "outside"

    # Save the current patient's safe-zone state
    patient_id = st.session_state.selected_patient_id

    st.session_state.patient_states[patient_id]["last_safe_zone_state"] = (
        st.session_state.last_safe_zone_state
    )


def check_medication_alerts(medications):

    for med in medications:

        if (
            med.get("active", True)
            and med["status"] == "Pending"
        ):

            message = (
                f"{patient['name']}: {med['name']} is scheduled for "
                f"{med['time']} and is still marked as pending."
            )

            # Check whether this medication alert already exists
            already_recorded = any(
                alert["type"] == "Medication Reminder"
                and alert["message"] == message
                for alert in st.session_state.alert_history
            )

            # Create an alert only if it hasn't been recorded before
            if not already_recorded:

                create_alert(
                    alert_type="Medication Reminder",
                    message=message,
                    priority="Medium"
                )


def check_gps_connection_alert(gps_status):

    previous_status = st.session_state.last_gps_status

    # GPS connection lost
    if gps_status == "Disconnected" and previous_status != "Disconnected":

        create_alert(
            alert_type="GPS Connection Lost",
            message=(
                "The patient's GPS connection has been lost. "
                "The caregiver cannot receive updated location information."
            ),
            priority="High"
        )

    # GPS connection restored
    elif gps_status == "Connected" and previous_status == "Disconnected":

        create_alert(
            alert_type="GPS Connection Restored",
            message=(
                "The patient's GPS connection has been restored. "
                "Location monitoring can continue."
            ),
            priority="Low"
        )

    # Remember the current status
    st.session_state.last_gps_status = gps_status

    # Save the status for the selected patient
    patient_id = st.session_state.selected_patient_id

    st.session_state.patient_states[patient_id]["last_gps_status"] = (
        gps_status
    )

def update_gps_status():
    patient_id = st.session_state.selected_patient_id

    widget_key = f"gps_status_selector_{patient_id}"

    new_status = st.session_state[widget_key]

    st.session_state.gps_connection_status = new_status

    st.session_state.patient_states[patient_id]["gps_connection_status"] = (
        new_status
    )
distance = calculate_distance(
    gps["latitude"],
    gps["longitude"],
    safe_zone["latitude"],
    safe_zone["longitude"]
)

is_inside = distance <= safe_zone["radius_m"]
# Automatically create a caregiver alert when necessary
if gps["status"] == "Connected":
    check_safe_zone_alert(
        is_inside,
        distance,
        safe_zone
    )


# ==========================================

# ==========================================
# 5. HEADER
# ==========================================

st.title("🧠 WANAS Dashboard")
st.caption("Patient Monitoring and Caregiver Management")

st.divider()



# ==========================================
# 6. PATIENT OVERVIEW
# ==========================================

if page == "Patient Overview":

    st.header("Patient Overview")

    # Patient Information
    col1, col2, col3 = st.columns(3)

    col1.metric("Patient Name", patient["name"])
    col2.metric("Age", patient["age"])
    col3.metric("Patient ID", patient["id"])

    st.divider()

    # Current System Status
    st.subheader("Current System Status")

    col1, col2, col3 = st.columns(3)

    if gps["status"] == "Connected":
        col1.success("GPS Connected")
    else:
        col1.error("GPS Unavailable")

    if is_inside:
        col2.success("Inside Safe Zone")
    else:
        col2.error("Outside Safe Zone")

    # Active medications only
    active_medications = [
        med for med in medications
        if med.get("active", True)
    ]

    pending_count = sum(
        1 for med in active_medications
        if med["status"] == "Pending"
    )

    col3.metric("Pending Medications", pending_count)

    st.divider()

    # Medication Statistics
    st.subheader("Medication Summary")

    total_active = len(active_medications)

    col1, col2 = st.columns(2)

    col1.metric("Active Medications", total_active)
    col2.metric("Pending Medications", pending_count)

    if active_medications:

        medication_table = pd.DataFrame(active_medications)

        st.dataframe(
            medication_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("This patient has no active medications.")

    st.divider()

    # Alert Summary
    st.subheader("Caregiver Alert Summary")

    patient_alerts = st.session_state.alert_history

    unacknowledged_count = sum(
        1 for alert in patient_alerts
        if alert["status"] == "Unacknowledged"
    )

    acknowledged_count = sum(
        1 for alert in patient_alerts
        if alert["status"] == "Acknowledged"
    )

    col1, col2 = st.columns(2)

    col1.metric("Unacknowledged Alerts", unacknowledged_count)
    col2.metric("Acknowledged Alerts", acknowledged_count)

    st.divider()

    # Recent Activity
    st.subheader("Recent Activity")

    if patient_alerts:

        recent_events = patient_alerts[:3]

        for event in recent_events:

            st.write(f"**{event['type']}**")
            st.write(event["message"])

            st.caption(event["timestamp"])

            st.divider()

    else:

        st.info("No recent activity recorded.")





# ==========================================
# 7. GPS AND SAFE ZONES
# ==========================================

elif page == "GPS & Safe Zones":

    st.header("GPS & Safe Zone Monitoring")

    st.info(
        "Simulation mode: Change the coordinates to test patient movement."
    )

    # Simulate GPS connection status
    gps["status"] = st.selectbox(
        "GPS Connection Status",
        options=["Connected", "Disconnected"],
        index=["Connected", "Disconnected"].index(
            st.session_state.gps_connection_status
        ),
        key=f"gps_status_selector_{patient_id}",
        on_change=update_gps_status,
        help="Use this dropdown to simulate GPS connection problems."
    )
    # Check for GPS connection changes
    check_gps_connection_alert(gps["status"])

    st.subheader("Simulate Patient Location")

    col1, col2 = st.columns(2)

    with col1:
        new_latitude = st.number_input(
            "Patient Latitude",
            value=float(st.session_state.patient_latitude),
            format="%.6f",
            key=f"latitude_input_{patient_id}"
        )

    with col2:
        new_longitude = st.number_input(
            "Patient Longitude",
            value=float(st.session_state.patient_longitude),
            format="%.6f",
            key=f"longitude_input_{patient_id}"
        )
    # Update simulated GPS location
    st.session_state.patient_latitude = new_latitude
    st.session_state.patient_longitude = new_longitude

    patient_state["patient_latitude"] = new_latitude
    patient_state["patient_longitude"] = new_longitude

    gps["latitude"] = new_latitude
    gps["longitude"] = new_longitude

    # --------------------------------------
    # CALCULATE DISTANCE
    # --------------------------------------

    distance = calculate_distance(
        gps["latitude"],
        gps["longitude"],
        safe_zone["latitude"],
        safe_zone["longitude"]
    )

    is_inside = distance <= safe_zone["radius_m"]
        # Automatically create an alert if the patient leaves the safe zone
    if gps["status"] == "Connected":
        check_safe_zone_alert(
            is_inside,
            distance,
            safe_zone
        )

    # --------------------------------------
    # DISPLAY GPS INFORMATION
    # --------------------------------------

    st.subheader("Current Location Status")

    col1, col2, col3 = st.columns(3)

    col1.metric("GPS Status", gps["status"])

    col2.metric(
        "Distance from Home",
        f"{distance:.1f} m"
    )

    col3.metric(
        "Safe Zone Radius",
        f'{safe_zone["radius_m"]} m'
    )

    if gps["status"] != "Connected":
        st.error("GPS location is unavailable.")

    elif is_inside:
        st.success("Patient is INSIDE the safe zone.")

    else:
        st.error("ALERT: Patient is OUTSIDE the safe zone!")

    # --------------------------------------
    # CREATE MAP
    # --------------------------------------

    st.subheader("Patient Location Map")

    map_data = pd.DataFrame({
        "Latitude": [
            gps["latitude"],
            safe_zone["latitude"]
        ],
        "Longitude": [
            gps["longitude"],
            safe_zone["longitude"]
        ],
        "Location": [
            "Patient",
            "Home / Safe Zone Center"
        ]
    })

    fig = px.scatter_map(
        map_data,
        lat="Latitude",
        lon="Longitude",
        color="Location",
        hover_name="Location",
        zoom=15,
        height=550,
        map_style="open-street-map",
        color_discrete_map={
            "Patient": "red",
            "Home / Safe Zone Center": "blue"
        }
    )

    # --------------------------------------
    # DRAW SAFE ZONE CIRCLE
    # --------------------------------------

    circle_latitudes = []
    circle_longitudes = []

    for angle in range(361):

        angle_rad = math.radians(angle)

        latitude_offset = (
            safe_zone["radius_m"] / 111320
        ) * math.cos(angle_rad)

        longitude_offset = (
            safe_zone["radius_m"] /
            (111320 * math.cos(math.radians(safe_zone["latitude"])))
        ) * math.sin(angle_rad)

        circle_latitudes.append(
            safe_zone["latitude"] + latitude_offset
        )

        circle_longitudes.append(
            safe_zone["longitude"] + longitude_offset
        )

    # Add filled safe-zone circle
    fig.add_trace(
        go.Scattermap(
            lat=circle_latitudes,
            lon=circle_longitudes,
            mode="lines",   
            name="Safe Zone Boundary",
            line=dict(width=3, color="green")
        )
    )

    # --------------------------------------
    # MAP CENTER AND DISPLAY
    # --------------------------------------

    fig.update_layout(
        map_style="open-street-map",
        map_center={
            "lat": safe_zone["latitude"],
            "lon": safe_zone["longitude"]
        },
        map_zoom=15,
        margin={
            "r": 0,
            "t": 0,
            "l": 0,
            "b": 0
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Red marker = Patient | Blue marker = Home | "
        "Green circle = Safe Zone Boundary"
    )

# ==========================================
# 8. MEDICATION TRACKING
# ==========================================

elif page == "Medication Tracking":

    st.header("Medication Tracking")

    st.write(
        f"Manage medication records for {patient['name']}."
    )

    if not any(med.get("active", True) for med in medications):
        st.info("There are no active medications for this patient.")
        active_medications = []
    else:
        active_medications = [
            (index, med)
            for index, med in enumerate(medications)
            if med.get("active", True)
        ]

    for index, med in active_medications:

        st.divider()

        st.subheader(med["name"])

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"Scheduled Time: {med['time']}")
            # Show who updated the medication
            if "reported_by" in med:
                st.write(f"Updated by: {med['reported_by']}")

            # Show when the medication was updated
            if "updated_at" in med:
                st.write(f"Last updated: {med['updated_at']}")

            current_status = med["status"]

            if current_status == "Taken (Reported)":
                current_status = "Taken"

            new_status = st.selectbox(
                "Medication Status",
                options=["Pending", "Taken", "Missed"],
                index=["Pending", "Taken", "Missed"].index(
                    current_status
                ),
                key=f"med_status_{patient_id}_{index}"
            )

        with col2:

            reported_by = st.selectbox(
                "Reported By",
                options=["Caregiver", "Patient"],
                key=f"med_reporter_{patient_id}_{index}"
            )

        if st.button(
            "Save Medication Update",
            key=f"save_med_{patient_id}_{index}"
        ):

            med["status"] = (
                "Taken (Reported)"
                if new_status == "Taken"
                else new_status
            )

            med["reported_by"] = reported_by

            med["updated_at"] = datetime.now().astimezone().strftime(
                "%Y-%m-%d %I:%M:%S %p"
            )

            # Permanently save the medication update
            save_patient_data()

            st.success(
                f"{med['name']} updated successfully!"
            )

            st.rerun()

    st.divider()

    st.caption(
        "Taken (Reported) means the patient or caregiver reported "
        "that the medication was taken. It does not confirm ingestion."
    )

# ==========================================
# 9. CAREGIVER ALERTS
# ==========================================


elif page == "Caregiver Alerts":

    st.header("Caregiver Alerts")
    # Check for pending medication alerts
    check_medication_alerts(medications)

    # Calculate the current patient location status
    alert_distance = calculate_distance(
        gps["latitude"],
        gps["longitude"],
        safe_zone["latitude"],
        safe_zone["longitude"]
    )

    alert_is_inside = alert_distance <= safe_zone["radius_m"]

    st.subheader("Current Patient Status")

    if gps["status"] != "Connected":

        st.error("GPS location is unavailable.")

    elif alert_is_inside:

        st.success("Patient is currently inside the safe zone.")

    else:

        st.error("ALERT: Patient is currently outside the safe zone!")

    # Medication alerts
    st.subheader("Medication Reminders")

    pending_medications = [
        med for med in medications
        if med["status"] == "Pending"
    ]

    if pending_medications:

        for med in pending_medications:
            st.warning(
                f"Medication reminder: {med['name']} "
                f"scheduled for {med['time']}."
            )

    else:

        st.success("There are no pending medications.")

    # Saved alert history
    st.divider()

    st.subheader("Alert History")

    if st.session_state.alert_history:

        # Count alerts that still need caregiver attention
        unacknowledged_count = sum(
            1 for alert in st.session_state.alert_history
            if alert["status"] == "Unacknowledged"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Alerts",
            len(st.session_state.alert_history)
        )

        col2.metric(
            "Unacknowledged Alerts",
            unacknowledged_count
        )

        st.divider()

        for alert in st.session_state.alert_history:

            col1, col2 = st.columns([3, 1])

            with col1:

                st.markdown(f"**{alert['type']}**")

                st.write(alert["message"])

                st.caption(
                    f"ID: {alert['id']} | "
                    f"Time: {alert['timestamp']}"
                )

            with col2:

                st.write("Priority")

                if alert["priority"] == "High":
                    st.error("HIGH")
                else:
                    st.info(alert["priority"])

                if alert["status"] == "Unacknowledged":

                    if st.button(
                        "Acknowledge",
                        key=f"ack_{alert['id']}"
                    ):
                        alert["status"] = "Acknowledged"

                        alert["acknowledged_by"] = "Caregiver"

                        alert["acknowledged_at"] = (
                            datetime.now().astimezone().strftime(
                                "%Y-%m-%d %I:%M:%S %p"
                            )
        )

                        save_alerts()
                        st.rerun()

                else:

                    st.success("Acknowledged")

    st.success("Acknowledged")

    if "acknowledged_by" in alert:
        st.caption(
            f"Acknowledged by: {alert['acknowledged_by']}"
        )

    if "acknowledged_at" in alert:
        st.caption(
            f"Date: {alert['acknowledged_at']}"
        )

        st.divider()

    else:

        st.info("No alerts have been recorded yet.")

# ==========================================
# 10. EVENT HISTORY
# ==========================================

elif page == "Event History":

    st.header("Event History")

    st.write(
        "A timeline of recorded patient safety alerts, "
        "medication reminders, and GPS connection events."
    )

    # Get all saved alerts
    events = st.session_state.alert_history

    if events:

        # Convert saved alerts into a DataFrame
        events_df = pd.DataFrame(events)
        # Add acknowledgment information if it is missing
        if "acknowledged_by" not in events_df.columns:
            events_df["acknowledged_by"] = "Not acknowledged"

        if "acknowledged_at" not in events_df.columns:
            events_df["acknowledged_at"] = "Not available"

        events_df["acknowledged_by"] = (
            events_df["acknowledged_by"].fillna("Not acknowledged")
        )

        events_df["acknowledged_at"] = (
            events_df["acknowledged_at"].fillna("Not available")
        )

        # Display summary
        total_events = len(events_df)

        acknowledged_events = sum(
            1 for event in events
            if event["status"] == "Acknowledged"
        )

        unacknowledged_events = sum(
            1 for event in events
            if event["status"] == "Unacknowledged"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Events", total_events)
        col2.metric("Acknowledged", acknowledged_events)
        col3.metric("Unacknowledged", unacknowledged_events)

        st.divider()
        # Search and Sort
        st.subheader("Search and Sort Events")

        col1, col2 = st.columns(2)

        with col1:
            search_query = st.text_input(
                "Search Events",
                placeholder="Search by ID, medication, or message..."
            )

        with col2:
            sort_order = st.selectbox(
                "Sort By Date",
                ["Newest First", "Oldest First"]
            )

        # Event History Filters
        st.subheader("Filter Events")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_type = st.selectbox(
                "Event Type",
                ["All"] + sorted(events_df["type"].unique().tolist())
            )

        with col2:
            selected_priority = st.selectbox(
                "Priority",
                ["All", "High", "Medium", "Low"]
            )

        with col3:
            selected_status = st.selectbox(
                "Status",
                ["All", "Acknowledged", "Unacknowledged"]
            )

        # Start with all events
        # Start with all events
        filtered_events = events_df.copy()

        # Search events
        if search_query.strip():

            search_text = search_query.strip().lower()

            filtered_events = filtered_events[
                filtered_events.astype(str).apply(
                    lambda column: column.str.lower().str.contains(
                        search_text,
                        na=False,
                        regex=False
                    )
                ).any(axis=1)
            ]

        # Sort events by date
        filtered_events = filtered_events.copy()

        filtered_events["_sort_time"] = pd.to_datetime(
            filtered_events["timestamp"],
            errors="coerce"
        )

        if sort_order == "Newest First":

            filtered_events = filtered_events.sort_values(
                "_sort_time",
                ascending=False,
                na_position="last"
            )

        else:

            filtered_events = filtered_events.sort_values(
                "_sort_time",
                ascending=True,
                na_position="last"
            )

        filtered_events = filtered_events.drop(
            columns=["_sort_time"]
        )

        # Filter by event type
        if selected_type != "All":
            filtered_events = filtered_events[
                filtered_events["type"] == selected_type
            ]

        # Filter by priority
        if selected_priority != "All":
            filtered_events = filtered_events[
                filtered_events["priority"] == selected_priority
            ]

        # Filter by status
        if selected_status != "All":
            filtered_events = filtered_events[
                filtered_events["status"] == selected_status
            ]

        # Display filtered results
        st.subheader("Recorded Events")

        st.dataframe(
            filtered_events[
                [
                    "id",
                    "timestamp",
                    "type",
                    "message",
                    "priority",
                    "status",
                    "acknowledged_by",
                    "acknowledged_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing {len(filtered_events)} "
            f"of {len(events_df)} events."
        )

    else:

        st.info("No events have been recorded yet.")

# ==========================================
# 11. ADD NEW PATIENT
# ==========================================

elif page == "Add New Patient":

    st.header("Add New Patient")

    st.write(
        "Enter the new patient's information below."
    )

    with st.form("add_patient_form"):

        # Patient information
        st.subheader("Patient Information")

        new_name = st.text_input("Patient Name")

        new_age = st.number_input(
            "Patient Age",
            min_value=1,
            max_value=120,
            value=70
        )

        # GPS information
        st.subheader("GPS Information")

        new_latitude = st.number_input(
            "GPS Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=31.2000,
            format="%.6f"
        )

        new_longitude = st.number_input(
            "GPS Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=29.9180,
            format="%.6f"
        )

        new_gps_status = st.selectbox(
            "GPS Connection Status",
            ["Connected", "Disconnected"]
        )

        # Safe zone information
        st.subheader("Safe Zone Information")

        new_safe_zone_name = st.text_input(
            "Safe Zone Name",
            value="Home"
        )

        new_radius = st.number_input(
            "Safe Zone Radius (meters)",
            min_value=50,
            max_value=5000,
            value=300
        )

        # Medication information
        st.subheader("First Medication")

        new_medication_name = st.text_input(
            "Medication Name"
        )

        new_medication_time = st.text_input(
            "Medication Time",
            value="09:00 AM"
        )

        submitted = st.form_submit_button(
            "Add Patient"
        )

        if submitted:

            if not new_name.strip():
                st.error("Please enter the patient's name.")

            elif any(
                p["name"].strip().lower() == new_name.strip().lower()
                for p in patients
            ):
                st.error("A patient with this name already exists.")

            elif not new_medication_name.strip():
                st.error("Please enter at least one medication.")

            else:

                new_patient_number = max(
                    int(p["id"].split("-")[1])
                    for p in patients
                ) + 1

                new_patient_id = (
                    f"WANAS-{new_patient_number:03d}"
                )

                new_patient = {
                    "id": new_patient_id,
                    "name": new_name.strip(),
                    "age": int(new_age),

                    "gps": {
                        "latitude": new_latitude,
                        "longitude": new_longitude,
                        "accuracy_m": 10,
                        "status": new_gps_status
                    },

                    "safe_zone": {
                        "name": new_safe_zone_name.strip() or "Home",
                        "latitude": new_latitude,
                        "longitude": new_longitude,
                        "radius_m": new_radius
                    },

                    "medications": [
                        {
                            "name": new_medication_name.strip(),
                            "time": new_medication_time.strip(),
                            "status": "Pending"
                        }
                    ]
                }

                patients.append(new_patient)

                save_patient_data()

                st.success(
                    f"{new_name} was added successfully! "
                    f"Patient ID: {new_patient_id}"
                )

                st.session_state.selected_patient_id = new_patient_id

                st.rerun()
# ==========================================
# 12. MANAGE MEDICATIONS
# ==========================================

elif page == "Manage Medications":

    st.header("Manage Medications")

    st.write(
        f"Add new medications for {patient['name']}."
    )

    st.subheader("Add a New Medication")

    with st.form("add_medication_form"):

        medication_name = st.text_input(
            "Medication Name"
        )

        medication_time = st.text_input(
            "Scheduled Time",
            placeholder="Example: 09:00 AM"
        )

        submitted = st.form_submit_button(
            "Add Medication"
        )

        if submitted:

            if not medication_name.strip():

                st.error("Please enter a medication name.")

            elif any(
                med["name"].strip().lower()
                == medication_name.strip().lower()
                for med in medications
            ):

                st.error(
                    "This medication already exists for this patient."
                )

            else:

                new_medication = {
                    "name": medication_name.strip(),
                    "time": medication_time.strip(),
                    "status": "Pending"
                }

                medications.append(new_medication)

                save_patient_data()

                st.success(
                    f"{medication_name} added successfully!"
                )

                st.rerun()

    st.divider()

    
    st.subheader("Current Medications")
    
    if medications:

        for index, med in enumerate(medications):

            st.divider()

            col1, col2 = st.columns([3, 1])

            with col1:

                st.write(f"**{med['name']}**")
                st.write(f"Scheduled Time: {med['time']}")

                # Older medications are Active by default
                if "active" not in med:
                    med["active"] = True

                if med["active"]:
                    st.success("Active")
                else:
                    st.warning("Inactive")

            with col2:

                if med["active"]:

                    if st.button(
                        "Deactivate",
                        key=f"deactivate_med_{patient_id}_{index}"
                    ):

                        med["active"] = False

                        save_patient_data()

                        st.success("Medication deactivated.")

                        st.rerun()

                else:

                    if st.button(
                        "Reactivate",
                        key=f"reactivate_med_{patient_id}_{index}"
                    ):

                        med["active"] = True

                        save_patient_data()

                        st.success("Medication reactivated.")

                        st.rerun()

    else:

        st.info("This patient has no medications yet.")
    
# ==========================================
# 13. EDIT PATIENT INFORMATION
# ==========================================

elif page == "Edit Patient Information":

    st.header("Edit Patient Information")

    st.write(
        f"Update the information for {patient['name']}."
    )

    with st.form("edit_patient_form"):

        # Patient Information
        st.subheader("Patient Information")

        edited_name = st.text_input(
            "Patient Name",
            value=patient["name"]
        )

        edited_age = st.number_input(
            "Patient Age",
            min_value=1,
            max_value=120,
            value=int(patient["age"])
        )

        # GPS Information
        st.subheader("GPS Information")

        edited_latitude = st.number_input(
            "GPS Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(gps["latitude"]),
            format="%.6f"
        )

        edited_longitude = st.number_input(
            "GPS Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(gps["longitude"]),
            format="%.6f"
        )

        # Safe Zone Information
        st.subheader("Safe Zone Information")

        edited_safe_zone_name = st.text_input(
            "Safe Zone Name",
            value=safe_zone["name"]
        )

        edited_radius = st.number_input(
            "Safe Zone Radius (meters)",
            min_value=50,
            max_value=5000,
            value=int(safe_zone["radius_m"])
        )

        submitted = st.form_submit_button(
            "Save Patient Changes"
        )

        if submitted:

            if not edited_name.strip():

                st.error("Patient name cannot be empty.")

            else:

                # Update patient information
                patient["name"] = edited_name.strip()
                patient["age"] = int(edited_age)

                # Update GPS coordinates
                gps["latitude"] = edited_latitude
                gps["longitude"] = edited_longitude

                # Update safe zone
                safe_zone["name"] = (
                    edited_safe_zone_name.strip() or "Home"
                )

                safe_zone["latitude"] = edited_latitude
                safe_zone["longitude"] = edited_longitude
                safe_zone["radius_m"] = edited_radius

                # Save changes permanently
                save_patient_data()

                st.success(
                    "Patient information updated successfully!"
                )

                st.rerun()




# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "WANAS Smart Care System | Demonstration Version"
)