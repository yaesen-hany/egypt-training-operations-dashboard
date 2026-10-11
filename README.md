# Insight Dashboard Builder

**Samsung Innovation Campus — Chapter 8: Streamlit, Data Visualization & AI-Assisted Dashboard Building**
**Day 2 · Units 41–42**

Build an interactive Streamlit dashboard using charts, maps, filters, and an AI-assisted iteration log.

---

## Scenario

- A program team wants to monitor training activity across cities and channels.
- The dashboard must help users explore performance, trends, geographic coverage, and key insights.
- Use Streamlit to turn raw tabular data into a clear, interactive product.

## Core Dataset Fields

| Field | Description |
|---|---|
| `date` | Date of the training activity |
| `city`, `governorate` | Location |
| `latitude`, `longitude` | Coordinates for mapping |
| `track` | e.g. AI, Data, Coding |
| `channel` | e.g. Online, Offline, Hybrid |
| `participants`, `completed` | Enrollment and completion counts |
| `satisfaction_score` | Participant satisfaction |
| `cost`, `revenue` | Financial values |

**Submission:** one Streamlit `.py` file + screenshots + AI prompt/refinement log.

---

## 1. Dataset & Data Preparation

Use or create a dataset with **at least 30 rows**. Each row represents one training activity or learning event.

- Load data from a CSV, or create a DataFrame inside the script.
- Convert the date column using `pd.to_datetime()`.
- Create at least one derived metric, such as `completion_rate` or `profit`.

**Sample rows**

```
date        | city       | track  | channel | participants | completed | satisfaction | cost  | revenue
2026-09-01  | Cairo      | AI     | Online  | 35           | 28        | 4.5          | 9000  | 16000
2026-09-03  | Giza       | Data   | Offline | 42           | 37        | 4.2          | 12000 | 21000
2026-09-05  | Alexandria | Coding | Hybrid  | 30           | 21        | 3.9          | 8000  | 14500
2026-09-07  | Aswan      | AI     | Offline | 24           | 19        | 4.7          | 7000  | 12500
```

**Required inspection**

- Show `df.head()`, `df.shape`, and a descriptive summary.
- Display the full dataset using `st.dataframe()`.
- Use `st.metric()` for at least three KPIs.

---

## 2. Dashboard Layout Requirements

### Sidebar Controls
- Date range filter
- Track selector
- City or governorate multiselect
- Channel selector
- Minimum satisfaction or completion-rate slider

### Main Dashboard
- Title, caption, and short markdown explanation
- At least 3 KPI metrics with deltas
- Use columns to organize KPIs and charts
- Use tabs for **Overview**, **Trends**, **Geography**, and **Data Explorer**

### Interactivity
- All charts must react to the selected filters
- At least one `st.data_editor()` or editable table
- One `st.download_button()` or exported filtered table
- Use `session_state` to remember at least one user preference

> The app should still run correctly when filters return a small subset of data.

---

## 3. Visualization Requirements

### Built-in Streamlit Charts
- `st.line_chart()` for a time trend
- `st.bar_chart()` for category comparison
- `st.area_chart()` for cumulative or multi-series activity

### Altair
- Create at least one declarative chart.
- Use `pd.melt()` if needed to reshape metrics.
- Map x, y, and color channels clearly.
- Render using `st.altair_chart()`.

### Plotly
- Create one interactive Plotly chart (pie/donut, stacked bar, or interactive line).
- Render using `st.plotly_chart()`.

### Map
- Use `st.map()` **or** a Folium map.
- Show city locations using latitude/longitude.
- For Folium: include popups or `CircleMarker`s and render through `st.components.v1.html()`.

---

## 4. AI-Assisted / Vibe Coding Requirement

You must submit an **AI iteration log** containing:

1. Initial prompt describing the dashboard goal and data fields
2. Generated code or a summary of the generated structure
3. Observed issue after running the app
4. Feedback prompt used to improve the result
5. Final changes you accepted, rejected, or edited manually

**Required final prompt structure**

```
Role: Streamlit dashboard developer
Goal: build an interactive dashboard for ...
Data columns: ...
Required charts: ...
Required filters: ...
Layout: sidebar + tabs + columns
Output: complete executable app.py
```

> AI can help generate code, but the final app must be tested, cleaned, and explained by the student.

---

## 5. Deliverables

- [ ] `app.py` file
- [ ] Dataset file or generated dataset code


## 6. Final Reflection

Answer briefly:

1. Which visualization best explains the data?
2. What issue did AI create or solve?
3. How did you verify the dashboard output?

---

## Submission Checklist

### Technical
- [ ] Dataset loads successfully
- [ ] Date converted with `pd.to_datetime()`
- [ ] Sidebar filters work together
- [ ] 3+ KPI metrics included
- [ ] `st.line_chart` / `st.bar_chart` / `st.area_chart` used
- [ ] Altair chart included
- [ ] Plotly chart included
- [ ] Map included using `st.map` or Folium

### Product
- [ ] `st.dataframe` or `st.data_editor` included
- [ ] Tabs and columns used properly
- [ ] Session State or cache used meaningfully
- [ ] No broken chart when filter output is small
- [ ] AI prompt log is attached
- [ ] README includes run command
- [ ] Screenshots are attached
- [ ] Final reflection answered

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
