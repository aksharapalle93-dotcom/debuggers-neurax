# FlowPilot

**Urban Traffic Flow & Incident Intelligence**  
**Team:** debuggers  
**Hackathon:** NeuraX Hackathon 3.0 — AI in Smart Cities

> A software-only decision-support system that helps traffic operators understand current road-network conditions, anticipate congestion 15–60 minutes ahead, choose safer short-term responses, and evaluate longer-term improvements.

---

## 1. Problem understanding

Large urban road networks can change within minutes. A slowdown on one segment can spill back into neighboring roads, while recurring bottlenecks may come from road geometry, limited capacity, peak-hour demand, road works, weather, events, or incidents. Traffic managers therefore need more than a map showing where congestion already exists.

The challenge requires a system that uses organizer-provided traffic and road-network data to:

1. Maintain an updated view of network conditions.
2. Identify congestion and abnormal traffic behavior.
3. Detect or classify incidents when the available data supports it.
4. Forecast traffic conditions 15–60 minutes ahead.
5. Recommend evidence-based diversions or operational responses.
6. Identify recurring bottlenecks and simulate possible network or infrastructure changes.
7. Communicate expected impact, confidence, and limitations clearly.

### Primary users

- **Traffic control-room operators:** need timely, explainable alerts and operational recommendations.
- **Traffic planners and municipal decision-makers:** need evidence for recurring bottlenecks and possible long-term improvements.

### Root problem

Traffic management is often reactive. Detecting a jam after it forms is not enough, and diverting traffic without checking downstream capacity can move congestion rather than reduce it. Long-term proposals can also be difficult to justify without measured before/after evidence.

### Our product insight

FlowPilot treats congestion as a changing network condition, not an isolated red point. It connects four stages:

**Observe --> detect --> forecast --> recommend and simulate**

Every alert or recommendation should show the supporting evidence, expected effect, confidence, and limitations.

### Scope and constraints

- Software-only prototype using organizer-provided datasets.
- All traffic responses, diversions, and infrastructure suggestions are advisory or simulated.
- No live signal control, camera access, GPS-device integration, roadside sensors, municipal infrastructure access, or construction work.
- This is not a commuter navigation application and not a generic chatbot.

---

## 2. Proposed architecture

```text
Organizer-provided datasets
          |
          v
Data ingestion and validation
(timestamp parsing, schema checks, missing-value flags)
          |
          v
Road-network representation + historical traffic store
          |
          +--------------------+
          |                    |
          v                    v
Baseline and anomaly      Traffic forecasting
analysis                  (15/30/45/60 min)
          |                    |
          +----------+---------+
                     v
          Decision-support engine
  (operational responses, diversion checks,
       confidence and supporting evidence)
                     |
          +----------+-----------+
          |                      |
          v                      v
What-if simulation       Streamlit interface
(recurring bottlenecks)  (map, alerts, forecasts,
                          recommendations, impact)
```

### Planned components

- **Data layer:** Pandas for loading, cleaning, time alignment, quality checks, and feature preparation.
- **Network layer:** NetworkX if usable road-topology fields are present. If topology is absent, the prototype will use the strongest available segment or corridor relationships and document that limitation.
- **Detection layer:** Per-location historical baselines plus persistence checks to distinguish recurring congestion from abnormal behavior and reduce false alarms.
- **Forecasting layer:** A time-aware regression pipeline using lagged traffic values, rolling statistics, time features, and neighboring-segment features when supported by the data.
- **Recommendation layer:** Rules and predicted network impact combined to rank feasible operational responses. A recommendation is withheld or marked low confidence when evidence is insufficient.
- **Simulation layer:** Replay selected historical conditions and compare a baseline scenario with a simulated operational or network change.
- **Interface layer:** Streamlit views for current state, forecasts, alerts, recommendations, confidence, and before/after comparisons.

### Technology stack

- Python
- Pandas and NumPy
- Scikit-learn
- NetworkX
- Streamlit
- Folium or PyDeck for mapping, depending on the available geographic fields
- Git and GitHub for version control and reproducibility

---

## 3. Approach

### Step 1 — Audit the data

Before selecting a model, we will document:

- Available files and columns
- Time resolution and date range
- Road-segment identifiers and network links
- Speed, flow, occupancy, travel-time, incident, weather, event, or road-work fields
- Missing values, duplicates, outliers, and inconsistent timestamps
- Whether ground-truth incident labels exist

Model selection will be based on the data that actually exists, not assumptions.

### Step 2 — Build road-specific baselines

A single global speed threshold is unsuitable because normal traffic differs by road, time, and day. We will estimate the normal condition for each segment or corridor using historical observations grouped by relevant time periods. Current observations will be compared with their own baseline.

### Step 3 — Detect congestion and abnormal behavior

The detection pipeline will combine:

- Deviation from the historical baseline
- Persistence across consecutive observations
- Supporting traffic variables available in the dataset
- Neighboring-segment behavior when network relationships are available

This is intended to control false alarms rather than flag every temporary slowdown as an incident.

### Step 4 — Forecast 15–60 minutes ahead

We will begin with a transparent baseline forecast, then compare it with a machine-learning model. Candidate features include:

- Recent traffic lags
- Rolling averages and trends
- Time-of-day and day-of-week
- Baseline deviation
- Upstream or neighboring-segment states, where available

Evaluation will use a chronological split: earlier periods for training and later, unseen periods for validation. Data will not be randomly shuffled across time.

### Step 5 — Generate explainable recommendations

For a detected or forecast problem, FlowPilot will present:

- What was detected or predicted
- Affected segment or corridor
- Evidence supporting the result
- Suggested operational response or diversion
- Expected effect based on the prototype model or simulation
- Confidence and known limitations

Diversion recommendations will consider alternative-route conditions where the road-network data supports this check.

### Step 6 — Simulate recurring-bottleneck interventions

For a selected recurring bottleneck, the prototype will compare the observed or baseline case with a clearly labeled simulated intervention. Possible interventions depend on the provided network data and may include changed route allocation, capacity assumptions, or corridor-level network modifications.

Simulation results will not be presented as real-world deployment outcomes. They will be labeled as prototype estimates.

### Step 7 — Test robustness and reliability

We plan to test:

- Unseen later time periods
- Missing-data scenarios
- Noisy traffic measurements
- Changed demand patterns when the dataset permits
- Low-confidence cases
- UI and pipeline failure handling

Reported metrics will be produced only after running the system on the organizer-provided data.

---

## 4. MVP workflow

The smallest complete demonstration will follow this flow:

```text
INPUT
Historical/current traffic and road-network data

INTELLIGENCE
Road-specific baseline comparison + abnormality detection + short-horizon forecast

ACTION
Ranked, evidence-based traffic-management recommendation

MEASURABLE RESULT
Predicted delay/congestion change and a before/after simulated comparison
```

### What judges will see

1. Select a timestamp or replay a historical traffic window.
2. View the network state and an abnormal segment or corridor.
3. Inspect why it was flagged against its learned baseline.
4. View the predicted state 15–60 minutes ahead.
5. Review a recommended response with confidence and reasoning.
6. Compare the baseline and simulated intervention outcomes.

---

## 5. Validation plan

Metrics will depend on the available labels and fields.

- **Congestion or incident detection:** precision, recall, F1 score, and false-alarm count when ground-truth labels exist.
- **Traffic forecasting:** MAE and RMSE for supported forecast horizons on unseen chronological data.
- **Recommendations:** feasibility checks, evidence shown, and estimated impact from the prototype simulation.
- **Robustness:** performance change under missing or noisy inputs.
- **Explainability:** baseline comparison, relevant inputs, confidence, and limitations displayed with each result.

No numerical performance claim will be added until it has been measured.

---

## 6. Current project status

- **Verified fact:** The official task requires software-only analysis of organizer-provided traffic and road-network datasets.
- **Verified fact:** Forecast horizons are 15–60 minutes, and all proposed actions must remain simulated or advisory.
- **Assumption pending data audit:** The supplied files contain stable timestamps and segment or corridor identifiers.
- **Assumption pending data audit:** Geographic coordinates or road connections are available for a network map.
- **Prototype simulation:** Before/after intervention impact will be estimated by the prototype, not claimed as a real deployment result.

The next milestone is to inspect the organizer-provided datasets and update this README with the confirmed schema, chosen features, final model, measured validation results, and reproducible run instructions.
