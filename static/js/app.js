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
const form = document.getElementById("analysisForm");
const progressBar = document.getElementById("progressBar");
const statusMessage = document.getElementById("statusMessage");
const resultsContainer = document.getElementById("resultsContainer");
const errorContainer = document.getElementById("errorContainer");
const progressSection = document.getElementById("progressSection");
const resultsSection = document.getElementById("resultsSection");
const submitButton = form?.querySelector('button[type="submit"]');

// Validate DOM elements
if (
  !form ||
  !progressBar ||
  !statusMessage ||
  !resultsContainer ||
  !progressSection ||
  !submitButton
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
    updateProgress(data);
  }
});

// Analysis results
socket.on("analysis_complete", (data) => {
  console.log("Analysis complete:", data);
  displayResults(data);
  enableForm();
  lastProgress = 0; // Reset progress for next analysis

  // Show completion message
  showToast("Analysis completed successfully!", "success");
});

// Helper functions
function updateProgress(data) {
  const statusMessage = document.getElementById("statusMessage");
  if (statusMessage) {
    statusMessage.textContent = data.message || "Processing...";
  }
}

function displayResults(data) {
  if (!resultsContainer) {
    console.error("Results container not found");
    return;
  }

  resultsContainer.innerHTML = "";

  // Create sections for each category
  const sections = {
    duplicates: createSection("Duplicate Images", "warning"),
    nature: createSection("Nature", "success"),
    abstract: createSection("Abstract", "primary"),
    anime: createSection("Anime & Cartoons", "info"),
    space: createSection("Space & Cosmic", "dark"),
    dark: createSection("Dark & Minimal", "secondary"),
    light: createSection("Light & Bright", "light"),
    art: createSection("Art & Design", "danger"),
    architecture: createSection("Architecture", "warning"),
    technology: createSection("Technology", "primary"),
    animals: createSection("Animals & Wildlife", "success"),
    other: createSection("Other", "secondary"),
  };

  // Process duplicates
  if (data.duplicates && Array.isArray(data.duplicates)) {
    const imageGrid = document.createElement("div");
    imageGrid.className = "row row-cols-1 row-cols-md-3 g-4 mt-2";
    imageGrid.style.display = "none"; // Initially hidden

    data.duplicates.forEach((group, groupIndex) => {
      group.forEach((path, index) => {
        const image = {
          path: path,
          filename: path.split("/").pop(),
          similar_to: index === 0 ? "Original" : `Duplicate ${index}`,
        };
        const imageCard = createImageCard(image, "duplicates");
        if (imageCard) {
          imageGrid.appendChild(imageCard);
        }
      });
    });

    // Add show/hide button
    const toggleButton = document.createElement("button");
    toggleButton.className = "btn btn-warning mb-3";
    toggleButton.textContent = "Show Duplicate Images";
    toggleButton.onclick = () => {
      if (imageGrid.style.display === "none") {
        imageGrid.style.display = "flex";
        toggleButton.textContent = "Hide Duplicate Images";
      } else {
        imageGrid.style.display = "none";
        toggleButton.textContent = "Show Duplicate Images";
      }
    };

    sections.duplicates.appendChild(toggleButton);
    sections.duplicates.appendChild(imageGrid);
    resultsContainer.appendChild(sections.duplicates);
  }

  // Process categories
  if (data.categories) {
    Object.entries(data.categories).forEach(([category, paths]) => {
      if (Array.isArray(paths) && paths.length > 0) {
        const imageGrid = document.createElement("div");
        imageGrid.className = "row row-cols-1 row-cols-md-3 g-4 mt-2";
        imageGrid.style.display = "none"; // Initially hidden

        paths.forEach((path) => {
          const image = {
            path: path,
            filename: path.split("/").pop(),
            quality_score: data.aesthetic_scores?.[path] || 0,
          };
          const imageCard = createImageCard(image, category);
          if (imageCard) {
            imageGrid.appendChild(imageCard);
          }
        });

        // Add show/hide button
        const toggleButton = document.createElement("button");
        toggleButton.className = `btn btn-${getCategoryColor(category)} mb-3`;
        toggleButton.textContent = `Show ${
          category.charAt(0).toUpperCase() + category.slice(1)
        } Images (${paths.length})`;
        toggleButton.onclick = () => {
          if (imageGrid.style.display === "none") {
            imageGrid.style.display = "flex";
            toggleButton.textContent = `Hide ${
              category.charAt(0).toUpperCase() + category.slice(1)
            } Images`;
          } else {
            imageGrid.style.display = "none";
            toggleButton.textContent = `Show ${
              category.charAt(0).toUpperCase() + category.slice(1)
            } Images (${paths.length})`;
          }
        };

        const section = sections[category] || createSection(category, "info");
        section.appendChild(toggleButton);
        section.appendChild(imageGrid);
        resultsContainer.appendChild(section);
      }
    });
  }

  // Show results section if it exists
  if (resultsSection) {
    resultsSection.classList.remove("d-none");
  }
}

function getCategoryColor(category) {
  const colors = {
    nature: "success",
    abstract: "primary",
    anime: "info",
    space: "dark",
    dark: "secondary",
    light: "light",
    art: "danger",
    architecture: "warning",
    technology: "primary",
    animals: "success",
    other: "secondary",
  };
  return colors[category] || "info";
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
  if (!path) {
    console.error("No path provided for deletion");
    showToast("Error: No image path provided", "error");
    return;
  }

  console.log("Attempting to delete image:", path);

  if (confirm("Are you sure you want to delete this image?")) {
    const encodedPath = encodeURIComponent(path);
    console.log("Encoded path:", encodedPath);

    fetch(`/api/delete/${encodedPath}`, {
      method: "DELETE",
    })
      .then((response) => {
        console.log("Delete response status:", response.status);
        if (!response.ok) {
          return response.json().then((err) => Promise.reject(err));
        }
        return response.json();
      })
      .then((data) => {
        console.log("Delete response data:", data);
        if (data.success) {
          showToast("Image deleted successfully", "success");
          // Remove the image card from the UI
          const card = document.querySelector(`[data-path="${path}"]`);
          if (card) {
            const col = card.closest(".col");
            if (col) {
              col.remove();
              console.log("Removed image card from UI");
            } else {
              console.warn("Could not find column element to remove");
            }
          } else {
            console.warn("Could not find card element to remove");
          }
        } else {
          showToast(data.error || "Failed to delete image", "error");
        }
      })
      .catch((error) => {
        console.error("Delete error:", error);
        showToast(error.error || "Error deleting image", "error");
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
  // Create toast container if it doesn't exist
  let container = document.getElementById("toastContainer");
  if (!container) {
    container = document.createElement("div");
    container.id = "toastContainer";
    container.className = "toast-container";
    document.body.appendChild(container);
  }

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

  container.appendChild(toast);

  const bsToast = new bootstrap.Toast(toast);
  bsToast.show();

  toast.addEventListener("hidden.bs.toast", () => {
    toast.remove();
  });
}

function disableForm() {
  if (!form || !submitButton) return;

  // Disable all form inputs
  form.querySelectorAll("input, select").forEach((el) => (el.disabled = true));
  submitButton.disabled = true;

  // Reset status
  if (statusMessage) statusMessage.textContent = "Starting analysis...";
  if (progressSection) progressSection.classList.remove("d-none");
}

function enableForm() {
  if (!form || !submitButton) return;

  // Enable all form inputs
  form.querySelectorAll("input, select").forEach((el) => (el.disabled = false));
  submitButton.disabled = false;
}

function clearResults() {
  if (resultsContainer) resultsContainer.innerHTML = "";
  if (errorContainer) {
    errorContainer.innerHTML = "";
    errorContainer.classList.add("d-none");
  }
  if (resultsSection) resultsSection.classList.add("d-none");
}

// Form submission
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    disableForm();
    clearResults();

    const formData = {
      directory: document.getElementById("directory")?.value || "",
      similarity_threshold: parseFloat(
        document.getElementById("similarity")?.value || "0.85"
      ),
      aesthetic_threshold: parseFloat(
        document.getElementById("threshold")?.value || "0.8"
      ),
      recursive: document.getElementById("recursive")?.checked || false,
      workers: parseInt(document.getElementById("workers")?.value || "16"),
      skip_duplicates:
        document.getElementById("skip_duplicates")?.checked || false,
      skip_aesthetics:
        document.getElementById("skip_aesthetics")?.checked || false,
      limit: parseInt(document.getElementById("limit")?.value || "0"),
    };

    // Validate directory path
    if (!formData.directory || !formData.directory.trim()) {
      showToast("Please enter a valid directory path", "error");
      enableForm();
      return;
    }

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
