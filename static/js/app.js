// Initialize Socket.IO with better configuration
const socket = io({
  transports: ["websocket", "polling"],
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  timeout: 60000,
  autoConnect: true,
  forceNew: true,
  pingTimeout: 60000,
  pingInterval: 25000,
});

// DOM Elements
const form = document.getElementById("analyzeForm");
const progressBar = document.getElementById("progressBar");
const statusMessage = document.getElementById("statusMessage");
const resultsContainer = document.getElementById("results");
const errorContainer = document.getElementById("errorContainer");
const progressSection = document.getElementById("progressSection");
const resultsSection = document.getElementById("resultsSection");

// Validate DOM elements
if (
  !form ||
  !progressBar ||
  !statusMessage ||
  !resultsContainer ||
  !errorContainer ||
  !progressSection ||
  !resultsSection
) {
  console.error("Required DOM elements not found");
  showToast(
    "Error: Required page elements not found. Please refresh the page.",
    "error"
  );
}

let lastProgress = 0;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// Socket.IO event handlers
socket.on("connect", () => {
  console.log("Connected to server");
  showToast("Connected to server", "success");
  reconnectAttempts = 0;
});

socket.on("connect_error", (error) => {
  console.error("Connection error:", error);
  reconnectAttempts++;
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    showToast(
      "Connection error: Maximum reconnection attempts reached. Please refresh the page.",
      "error"
    );
    socket.disconnect();
  } else {
    showToast(
      `Connection error: ${error.message}. Attempting to reconnect...`,
      "warning"
    );
  }
});

socket.on("disconnect", (reason) => {
  console.log("Disconnected:", reason);
  if (reason === "io server disconnect") {
    // Server initiated disconnect, try to reconnect
    socket.connect();
  }
  showToast("Disconnected from server: " + reason, "warning");
});

socket.on("reconnect_attempt", (attemptNumber) => {
  console.log("Reconnection attempt:", attemptNumber);
  showToast(
    `Attempting to reconnect (${attemptNumber}/${MAX_RECONNECT_ATTEMPTS})...`,
    "info"
  );
});

socket.on("reconnect_failed", () => {
  console.error("Failed to reconnect");
  showToast("Failed to reconnect to server. Please refresh the page.", "error");
});

socket.on("error", (error) => {
  console.error("Socket error:", error);
  showToast("Error: " + error.message, "error");
});

// Progress updates with state preservation
socket.on("progress", (data) => {
  if (data.progress >= lastProgress) {
    lastProgress = data.progress;
    updateProgress(data.progress, data.message);
    showDetailedProgress(data);
  }
});

// Analysis results
socket.on("analysis_complete", (data) => {
  console.log("Analysis complete:", data);
  displayResults(data);
  document.getElementById("submitBtn").disabled = false;
  lastProgress = 0; // Reset progress for next analysis

  // Show completion message
  showToast("Analysis completed successfully!", "success");

  // Update final progress details
  showDetailedProgress(data);
});

// Form submission
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    disableForm();
    clearResults();

    const formData = {
      directory: document.getElementById("directory")?.value || "",
      similarity_threshold: parseFloat(
        document.getElementById("similarity")?.value || "0.9"
      ),
      aesthetic_threshold: parseFloat(
        document.getElementById("threshold")?.value || "0.5"
      ),
      recursive: document.getElementById("recursive")?.checked || false,
      workers: parseInt(document.getElementById("workers")?.value || "4"),
      skip_duplicates:
        document.getElementById("skip_duplicates")?.checked || false,
      skip_aesthetics:
        document.getElementById("skip_aesthetics")?.checked || false,
      limit: parseInt(document.getElementById("limit")?.value || "0"),
    };

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Analysis failed");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      showToast(error.message, "error");
      enableForm();
    }
  });
}

// Helper functions
function updateProgress(progress, message) {
  if (progressBar) {
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute("aria-valuenow", progress);
    progressBar.textContent = `${progress.toFixed(1)}%`;
  }

  if (statusMessage && message) {
    // Create a more detailed status message with timestamp
    const timestamp = new Date().toLocaleTimeString();
    const statusHTML = `
      <div class="d-flex justify-content-between align-items-center">
        <span>${message}</span>
        <small class="text-muted">${timestamp}</small>
      </div>
    `;
    statusMessage.innerHTML = statusHTML;
    console.log(
      `[${timestamp}] Progress: ${progress.toFixed(1)}% - ${message}`
    );
  }
}

function displayResults(data) {
  if (!resultsContainer || !resultsSection) return;

  resultsContainer.innerHTML = "";

  // Create sections for each category
  const sections = {
    duplicates: createSection("Duplicate Images", "warning"),
    low_quality: createSection("Low Quality Images", "danger"),
    high_quality: createSection("High Quality Images", "success"),
    uncategorized: createSection("Uncategorized Images", "info"),
  };

  // Process duplicates
  if (data.duplicates && Array.isArray(data.duplicates)) {
    const imageGrid = document.createElement("div");
    imageGrid.className = "row row-cols-1 row-cols-md-3 g-4 mt-2";

    data.duplicates.forEach((group, groupIndex) => {
      group.forEach((path, index) => {
        const image = {
          path: path,
          filename: path.split("/").pop(),
          similar_to: index === 0 ? "Original" : `Duplicate ${index}`,
        };
        const imageCard = createImageCard(image, "duplicates");
        imageGrid.appendChild(imageCard);
      });
    });

    sections.duplicates.appendChild(imageGrid);
    resultsContainer.appendChild(sections.duplicates);
  }

  // Process categories
  if (data.categories) {
    Object.entries(data.categories).forEach(([category, paths]) => {
      if (Array.isArray(paths) && paths.length > 0) {
        const imageGrid = document.createElement("div");
        imageGrid.className = "row row-cols-1 row-cols-md-3 g-4 mt-2";

        paths.forEach((path) => {
          const image = {
            path: path,
            filename: path.split("/").pop(),
            quality_score: data.aesthetic_scores?.[path] || 0,
          };
          const imageCard = createImageCard(image, category);
          imageGrid.appendChild(imageCard);
        });

        const section = sections[category] || createSection(category, "info");
        section.appendChild(imageGrid);
        resultsContainer.appendChild(section);
      }
    });
  }

  // Show results section
  resultsSection.classList.remove("d-none");
}

// Create a section container
function createSection(title, type) {
  const section = document.createElement("div");
  section.className = "mb-4";

  const header = document.createElement("h3");
  header.className = `text-${type} mb-3`;
  header.textContent = title;

  section.appendChild(header);
  return section;
}

// Create an image card
function createImageCard(image, category) {
  if (!image || !image.path) {
    console.error("Invalid image data:", image);
    return null;
  }

  const col = document.createElement("div");
  col.className = "col";

  const card = document.createElement("div");
  card.className = "card h-100";
  card.setAttribute("data-path", image.path);

  const img = document.createElement("img");
  img.src = `/api/image?path=${encodeURIComponent(image.path)}`;
  img.className = "card-img-top";
  img.style.height = "200px";
  img.style.objectFit = "cover";
  img.onerror = () => {
    img.src = "/static/placeholder.png";
    img.alt = "Image not found";
  };

  const cardBody = document.createElement("div");
  cardBody.className = "card-body";

  const title = document.createElement("h5");
  title.className = "card-title";
  title.textContent = image.filename || image.path.split("/").pop();

  const details = document.createElement("p");
  details.className = "card-text";

  if (category === "duplicates" && image.similar_to) {
    details.textContent = `Similar to: ${image.similar_to}`;
  } else if (
    (category === "low_quality" || category === "high_quality") &&
    image.quality_score !== undefined
  ) {
    details.textContent = `Quality Score: ${image.quality_score.toFixed(2)}`;
  }

  const actions = document.createElement("div");
  actions.className = "btn-group mt-2";

  if (category === "duplicates" || category === "low_quality") {
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "btn btn-danger btn-sm";
    deleteBtn.textContent = "Delete";
    deleteBtn.onclick = () => deleteImage(image.path);
    actions.appendChild(deleteBtn);
  }

  if (category === "uncategorized") {
    const organizeBtn = document.createElement("button");
    organizeBtn.className = "btn btn-primary btn-sm";
    organizeBtn.textContent = "Organize";
    organizeBtn.onclick = () => organizeImage(image.path);
    actions.appendChild(organizeBtn);
  }

  cardBody.appendChild(title);
  cardBody.appendChild(details);
  cardBody.appendChild(actions);

  card.appendChild(img);
  card.appendChild(cardBody);
  col.appendChild(card);

  return col;
}

// Delete an image
function deleteImage(path) {
  if (confirm("Are you sure you want to delete this image?")) {
    fetch(`/delete/${encodeURIComponent(path)}`, {
      method: "DELETE",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          showToast("Image deleted successfully", "success");
          // Remove the image card from the UI
          const card = document.querySelector(`[data-path="${path}"]`);
          if (card) {
            card.remove();
          }
        } else {
          showToast("Failed to delete image", "error");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        showToast("Error deleting image", "error");
      });
  }
}

// Organize an image
function organizeImage(path) {
  fetch(`/organize/${encodeURIComponent(path)}`, {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showToast("Image organized successfully", "success");
        // Remove the image card from the UI
        const card = document.querySelector(`[data-path="${path}"]`);
        if (card) {
          card.remove();
        }
      } else {
        showToast("Failed to organize image", "error");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      showToast("Error organizing image", "error");
    });
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast align-items-center text-white bg-${type} border-0`;
  toast.setAttribute("role", "alert");
  toast.setAttribute("aria-live", "assertive");
  toast.setAttribute("aria-atomic", "true");

  toast.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">
        ${message}
      </div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;

  const container = document.getElementById("toastContainer");
  container.appendChild(toast);

  const bsToast = new bootstrap.Toast(toast);
  bsToast.show();

  toast.addEventListener("hidden.bs.toast", () => {
    toast.remove();
  });
}

function disableForm() {
  if (!form) return;
  form.querySelectorAll("input, button").forEach((el) => (el.disabled = true));
  if (progressBar) progressBar.style.width = "0%";
  if (statusMessage) statusMessage.textContent = "Starting analysis...";
  if (progressSection) progressSection.classList.remove("d-none");
}

function enableForm() {
  if (!form) return;
  form.querySelectorAll("input, button").forEach((el) => (el.disabled = false));
}

function clearResults() {
  if (resultsContainer) resultsContainer.innerHTML = "";
  if (errorContainer) {
    errorContainer.innerHTML = "";
    errorContainer.classList.add("d-none");
  }
  if (resultsSection) resultsSection.classList.add("d-none");
}

// Add a new function to show detailed progress
function showDetailedProgress(data) {
  if (!progressSection) return;

  // Create or update the detailed progress section
  let detailsContainer = document.getElementById("progressDetails");
  if (!detailsContainer) {
    detailsContainer = document.createElement("div");
    detailsContainer.id = "progressDetails";
    detailsContainer.className = "mt-3";
    progressSection.appendChild(detailsContainer);
  }

  // Update the details
  detailsContainer.innerHTML = `
    <div class="card">
      <div class="card-body">
        <h6 class="card-subtitle mb-2 text-muted">Analysis Details</h6>
        <div class="row">
          <div class="col-md-6">
            <p class="mb-1"><strong>Total Images:</strong> ${
              data.total_images || 0
            }</p>
            <p class="mb-1"><strong>Processed:</strong> ${
              data.processed_images || 0
            }</p>
          </div>
          <div class="col-md-6">
            <p class="mb-1"><strong>Duplicates Found:</strong> ${
              data.duplicates?.length || 0
            }</p>
            <p class="mb-1"><strong>Categories:</strong> ${
              Object.keys(data.categories || {}).length
            }</p>
          </div>
        </div>
      </div>
    </div>
  `;
}
