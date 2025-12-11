// Simple enum-like role definition
const UserRole = {
  Admin: "Admin",
  CourseCreator: "CourseCreator",
};

let currentRole = UserRole.Admin;
let telemetryData = null;
let isLoading = false;

// --- Telemetry mapping logic (port of telemetry_mapper.ts) ---
function mapTelemetryForAdmin(data) {
  if (!data) return null;
  return {
    errorGraphs: data.feedback && data.feedback.failure_trends,
    performanceHeatmaps: data.performance,
    technicalQueue: {
      activeTasks: data.workload?.active_tasks ?? 0,
      queueDepth: data.workload?.max_queue_depth ?? 0,
      taskLog: data.workload?.task_log ?? [],
    },
  };
}

function mapTelemetryForCreator(data) {
  if (!data) return null;

  const totalEvents = data.feedback?.total_events ?? 0;
  const activeTasks = data.workload?.active_tasks ?? 0;

  const percent = activeTasks > 0 ? 50 : 100;
  const status =
    activeTasks > 0
      ? `Processing ${activeTasks} tasks...`
      : totalEvents > 0
      ? "Completed"
      : "Ready";

  const waitTimeMinutes = activeTasks * 2;
  const estimatedWaitTime =
    activeTasks > 0 ? `~${waitTimeMinutes} mins` : "Ready";

  const alerts = [];
  const recentFailures = data.feedback?.recent_failures || [];
  for (const failure of recentFailures) {
    if (failure.stage && failure.stage.includes("review")) {
      alerts.push(
        "Please refine your learning outcomes to improve content quality."
      );
    }
  }
  const uniqueAlerts = [...new Set(alerts)];

  return {
    progressBar: {
      percent,
      status,
    },
    estimatedWaitTime,
    smartAlerts: uniqueAlerts.length ? uniqueAlerts : undefined,
  };
}

function mapTelemetry(rawTelemetry, role) {
  if (role === UserRole.Admin) {
    return mapTelemetryForAdmin(rawTelemetry);
  }
  if (role === UserRole.CourseCreator) {
    return mapTelemetryForCreator(rawTelemetry);
  }
  return null;
}

// --- Rendering helpers ---
function setLoading(loading) {
  isLoading = loading;
  const loadingEl = document.getElementById("loading");
  const submitBtn = document.getElementById("submit-btn");
  const emptyState = document.getElementById("empty-state");

  if (loadingEl) loadingEl.style.display = loading ? "block" : "none";
  if (submitBtn) submitBtn.disabled = loading;
  if (submitBtn)
    submitBtn.textContent = loading ? "Generating..." : "Generate Course";
  if (emptyState && loading) emptyState.style.display = "none";
}

function render() {
  const contentEl = document.getElementById("content");
  const emptyState = document.getElementById("empty-state");

  if (!contentEl) return;

  // Clear previous content
  contentEl.innerHTML = "";

  if (isLoading) {
    return;
  }

  if (!telemetryData) {
    if (emptyState) emptyState.style.display = "block";
    return;
  }

  if (emptyState) emptyState.style.display = "none";

  const mapped = mapTelemetry(telemetryData, currentRole);
  if (!mapped) {
    contentEl.innerHTML = "<div class='panel'>Access Denied</div>";
    return;
  }

  if (currentRole === UserRole.Admin) {
    renderAdminDashboard(contentEl, mapped);
  } else if (currentRole === UserRole.CourseCreator) {
    renderCreatorStatus(contentEl, mapped);
  }
}

function renderAdminDashboard(container, view) {
  const panel = document.createElement("div");
  panel.className = "panel";

  const header = document.createElement("div");
  header.innerHTML = "<h2>Admin Telemetry Dashboard</h2>";
  panel.appendChild(header);

  // Failure trends
  const failureSection = document.createElement("div");
  failureSection.innerHTML =
    "<div style='font-weight:bold;margin-bottom:10px;'>Failure Trends (Rejections per Stage)</div>";
  if (view.errorGraphs && Object.keys(view.errorGraphs).length > 0) {
    const chart = document.createElement("div");
    chart.className = "bar-chart";
    Object.entries(view.errorGraphs).forEach(([stage, count]) => {
      const bar = document.createElement("div");
      bar.className = "bar";
      const height = Math.min(Number(count) * 20, 100);
      bar.style.height = `${height}%`;
      bar.title = `${stage}: ${count}`;
      chart.appendChild(bar);
    });
    failureSection.appendChild(chart);
  } else {
    const noFail = document.createElement("div");
    noFail.textContent = "No failures recorded.";
    failureSection.appendChild(noFail);
  }
  panel.appendChild(failureSection);

  // Performance table
  const perfSection = document.createElement("div");
  perfSection.innerHTML =
    "<div style='font-weight:bold;margin:15px 0 10px;'>Performance Metrics</div>";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  thead.innerHTML =
    "<tr><th>Stage</th><th>Avg (s)</th><th>Max (s)</th><th>Count</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  Object.entries(view.performanceHeatmaps || {}).forEach(
    ([stage, stats]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${stage}</td>
        <td>${Number(stats.avg_seconds).toFixed(2)}</td>
        <td>${Number(stats.max_seconds).toFixed(2)}</td>
        <td>${stats.count}</td>
      `;
      tbody.appendChild(tr);
    }
  );
  table.appendChild(tbody);
  perfSection.appendChild(table);
  panel.appendChild(perfSection);

  // Technical queue
  const queueSection = document.createElement("div");
  queueSection.innerHTML =
    "<div style='font-weight:bold;margin:15px 0 10px;'>Workload Queue</div>";

  const cards = document.createElement("div");
  cards.className = "card-grid";
  const activeCard = document.createElement("div");
  activeCard.className = "stat-card";
  activeCard.innerHTML = `
    <div>Active Tasks</div>
    <div class="stat-value">${view.technicalQueue.activeTasks}</div>
  `;
  const depthCard = document.createElement("div");
  depthCard.className = "stat-card";
  depthCard.innerHTML = `
    <div>Max Depth</div>
    <div class="stat-value">${view.technicalQueue.queueDepth}</div>
  `;
  cards.appendChild(activeCard);
  cards.appendChild(depthCard);
  queueSection.appendChild(cards);

  const recentTitle = document.createElement("h4");
  recentTitle.textContent = "Recent Tasks";
  queueSection.appendChild(recentTitle);

  const list = document.createElement("ul");
  list.style.fontSize = "0.9em";
  list.style.color = "#666";
  (view.technicalQueue.taskLog || []).slice(-5).forEach((task) => {
    const li = document.createElement("li");
    const dur =
      typeof task.duration === "number"
        ? task.duration.toFixed(2)
        : String(task.duration || "?");
    li.textContent = `${task.task_name} - ${
      task.success ? "✅" : "❌"
    } (${dur}s)`;
    list.appendChild(li);
  });
  queueSection.appendChild(list);
  panel.appendChild(queueSection);

  container.appendChild(panel);
}

function renderCreatorStatus(container, view) {
  const panel = document.createElement("div");
  panel.className = "panel";

  const header = document.createElement("div");
  header.innerHTML = "<h2>Course Generation Status</h2>";
  panel.appendChild(header);

  // Progress
  const progressSection = document.createElement("div");
  progressSection.style.marginBottom = "20px";

  const headerRow = document.createElement("div");
  headerRow.style.display = "flex";
  headerRow.style.justifyContent = "space-between";
  headerRow.style.marginBottom = "5px";
  headerRow.innerHTML = `<strong>${view.progressBar?.status || ""}</strong>
    <span>${view.progressBar?.percent ?? 0}%</span>`;
  progressSection.appendChild(headerRow);

  const progressContainer = document.createElement("div");
  progressContainer.className = "progress-container";
  const progressFill = document.createElement("div");
  progressFill.className = "progress-fill";
  progressFill.style.width = `${view.progressBar?.percent ?? 0}%`;
  progressContainer.appendChild(progressFill);
  progressSection.appendChild(progressContainer);
  panel.appendChild(progressSection);

  // Wait time
  const waitSection = document.createElement("div");
  waitSection.style.marginBottom = "20px";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = `⏱ Estimated Wait: ${view.estimatedWaitTime || "N/A"}`;
  waitSection.appendChild(badge);
  panel.appendChild(waitSection);

  // Smart alerts
  if (view.smartAlerts && view.smartAlerts.length) {
    const alertBox = document.createElement("div");
    alertBox.className = "alert-box";
    const strong = document.createElement("strong");
    strong.textContent = "💡 Suggestions:";
    alertBox.appendChild(strong);
    const ul = document.createElement("ul");
    ul.style.margin = "5px 0 0 20px";
    (view.smartAlerts || []).forEach((msg) => {
      const li = document.createElement("li");
      li.textContent = msg;
      ul.appendChild(li);
    });
    alertBox.appendChild(ul);
    panel.appendChild(alertBox);
  }

  container.appendChild(panel);
}

// --- Event wiring ---
function setup() {
  const form = document.getElementById("course-form");
  const roleToggle = document.getElementById("role-toggle");
  const roleBadge = document.getElementById("role-badge");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const titleInput = document.getElementById("course-title");
      const outcomesInput = document.getElementById("learning-outcomes");
      if (!titleInput || !outcomesInput) return;

      const courseTitle = titleInput.value;
      const learningOutcomes = outcomesInput.value
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);

      setLoading(true);
      telemetryData = null;
      render();

      try {
        const res = await fetch("/generate-course", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            course_title: courseTitle,
            learning_outcomes: learningOutcomes,
            skip_video: true,
            num_modules: 1,
            num_lessons: 2,
          }),
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`API Error ${res.status}: ${text}`);
        }

        const json = await res.json();
        if (json && json.telemetry) {
          telemetryData = json.telemetry;
        } else {
          telemetryData = null;
          console.warn("No telemetry field in response");
        }
      } catch (err) {
        console.error("Failed to generate course:", err);
        alert("Failed to generate course. Check console for details.");
      } finally {
        setLoading(false);
        render();
      }
    });
  }

  if (roleToggle && roleBadge) {
    roleToggle.addEventListener("click", () => {
      currentRole =
        currentRole === UserRole.Admin
          ? UserRole.CourseCreator
          : UserRole.Admin;
      roleBadge.textContent = `Current View: ${currentRole}`;
      roleToggle.textContent =
        "Switch to " +
        (currentRole === UserRole.Admin ? "Creator" : "Admin") +
        " View";
      render();
    });
  }

  render();
}

document.addEventListener("DOMContentLoaded", setup);
