# Frontend Component Specifications

Detailed specifications for each React UI component used in the tactical research console dashboard.

---

## 🧭 HeroSection

Displays active systems status indicators and executes a CSS-based animation illustrating feature flow between pipeline stages.

*   **File Path**: `webapp/frontend/src/components/HeroSection.jsx`
*   **Aesthetic Theme**: Industrial, sci-fi HUD frame with diagonal connector lines.

### Component Definition:
```javascript
function HeroSection()
```

---

## 🎛️ ResearchConsole

Handles multipart file selection drop zones and exposes slider controls for pipeline parameters.

*   **File Path**: `webapp/frontend/src/components/ResearchConsole.jsx`

### Properties:
| Prop Name | Type | Description |
| :--- | :--- | :--- |
| `imageFile` | `File \| null` | Currently active image target. |
| `params` | `Object` | Map of key-value parameters. |
| `onFileChange` | `Function` | Triggered when a new file is dropped or selected. |
| `onParamChange`| `Function` | Triggered when a parameter slider is dragged. |
| `onExecute` | `Function` | Submits form content requests to the backend. |
| `loading` | `Boolean` | Disables controls during active processing. |
| `error` | `String \| null` | Text showing validation or server error alerts. |

---

## 🖼️ ResultsPanel

Visualizer tabbed canvas showing base64 outputs. Includes a toggleable HTML overlay that dynamically draws bounding boxes over the raw original image at original scaling levels.

*   **File Path**: `webapp/frontend/src/components/ResultsPanel.jsx`

### Properties:
| Prop Name | Type | Description |
| :--- | :--- | :--- |
| `result` | `Object \| null` | The API response object. |
| `loading` | `Boolean` | Toggles skeletal loaders. |

---

## 📊 MetricsPanel

Exposes pipeline execution summaries and telemetry details using incremental animated counters.

*   **File Path**: `webapp/frontend/src/components/MetricsPanel.jsx`

### Properties:
| Prop Name | Type | Description |
| :--- | :--- | :--- |
| `result` | `Object \| null` | The API response object. |

---

## 📈 PipelineGraph

Renders an interactive stage description list. Clicking or hovering over any stage reveals details regarding the underlying models and layers involved.

*   **File Path**: `webapp/frontend/src/components/PipelineGraph.jsx`
