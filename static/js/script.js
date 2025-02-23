// Ensure DOM is loaded before running the script
document.addEventListener("DOMContentLoaded", () => {
  const menuButton = document.getElementById("menuButton");
  const fullscreenMenu = document.getElementById("fullscreenMenu");
  const closeButton = document.getElementById("closeButton");

  // Open the fullscreen menu
  menuButton?.addEventListener("click", () => {
    fullscreenMenu.classList.add("show");
  });

  // Close the fullscreen menu
  closeButton?.addEventListener("click", () => {
    fullscreenMenu.classList.remove("show");
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const tabs = {
    searchCoursesTab: document.getElementById("searchCoursesTab"),
    myCoursesTab: document.getElementById("myCoursesTab"),
    allCoursesTab: document.getElementById("allCoursesTab"),
    generalTab: document.getElementById("generalTab"),
    myPreferencesTab: document.getElementById("myPreferencesTab"),
  };

  const contents = {
    searchCoursesForm: document.getElementById("searchCoursesForm"),
    recommendationContent: document.getElementById("recommendationContent"),
    allCoursesContent: document.getElementById("allCoursesContent"),
    myCoursesContent: document.getElementById("myCoursesContent"),
    generalContent: document.getElementById("generalTabContent"),
    myPreferencesContent: document.getElementById("myPreferencesContent"),
    fillPreferencesForm: document.getElementById("fillPreferences"),
  };

  /**
   * Activate a tab and show corresponding content
   * @param {string} activeTabId - The ID of the tab to activate
   */

  const activateTab = (activeTabId) => {
    // Ensure the tab exists before attempting operations
    if (!tabs[activeTabId]) return;

    // Deactivate all tabs and hide all content
    Object.values(tabs).forEach((tab) => tab?.classList.remove("active"));
    Object.values(contents).forEach((content) => {
      if (content) content.style.display = "none";
    });

    // Activate the selected tab and display its corresponding content (only if they exist)
    tabs[activeTabId]?.classList.add("active");

    if (
      activeTabId == "searchCoursesTab" &&
      contents.searchCoursesForm &&
      contents.recommendationContent
    ) {
      if (
        typeof available_recommendations !== "undefined" &&
        available_recommendations === "True"
      ) {
        contents.searchCoursesForm.style.display = "none";

        contents.recommendationContent.style.display = "flex";
      } else {
        contents.searchCoursesForm.style.display = "block";
        contents.recommendationContent.style.display = "none";
      }
    } else if (activeTabId === "myCoursesTab" && contents.myCoursesContent) {
      contents.myCoursesContent.style.display = "flex";
      contents.fillPreferencesForm.style.display = "none";
    } else if (activeTabId === "allCoursesTab" && contents.allCoursesContent) {
      contents.allCoursesContent.style.display = "flex";
    } else if (activeTabId === "generalTab" && contents.generalContent) {
      contents.generalContent.style.display = "block";
      contents.myPreferencesContent.style.display = "none";
    } else if (
      activeTabId === "myPreferencesTab" &&
      contents.myPreferencesContent
    ) {
      contents.myPreferencesContent.style.display = "flex";
      contents.generalContent.style.display = "none";
    } else if (activeTabId === "searchCoursesTab") {
      contents.myPreferencesContent.style.display = "flex";
      // contents.generalContent.style.display = "none";
    }
  };

  // Check if the activeTab variable exists and is valid
  if (typeof activeTab !== "undefined" && tabs[activeTab]) {
    activateTab(activeTab);
  }

  // Add event listeners to tabs that exist
  Object.keys(tabs).forEach((tabId) => {
    if (tabs[tabId]) {
      tabs[tabId].addEventListener("click", () => activateTab(tabId));
    }
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const links = document.querySelectorAll(".sidebar nav a");

  links.forEach((link) => {
    if (window.location.href.includes(link.href.split("/")[3])) {
      link.classList.add("active");
    }
  });
});

document.addEventListener("DOMContentLoaded", () => {
  searchCoursesForm = document.getElementById("searchCoursesForm");
  recommendationContent = document.getElementById("recommendationContent");

  if (
    typeof available_recommendations !== "undefined" &&
    available_recommendations == "True"
  ) {
    searchCoursesForm.style.display = "none";
    recommendationContent.style.display = "flex";
  } else if (searchCoursesForm || recommendationContent) {
    searchCoursesForm.style.display = "block";
    recommendationContent.style.display = "none";
  }

  if (typeof error !== "undefined" && error == "course_add_clash") {
    alert("This course has already been added to your your courses");
  }
});

window.onload = function () {
  // Hide all programmatic horizontal select boxes when the page loads
  var programmingLevels = document.querySelectorAll(".programming-level");
  programmingLevels.forEach(function (levelDiv) {
    levelDiv.style.display = "none";
  });
  var languageLevels = document.querySelectorAll(".language-level");
  languageLevels.forEach(function (levelDiv) {
    levelDiv.style.display = "none";
  });
};
function showLanguageLevel(event) {
  var checkbox = event.target;
  // Get the parent element of the checkbox
  var parentDiv = checkbox.parentNode;
  // Find child elements with the language-level class in the parent element
  var languageLevelDiv = parentDiv.querySelector(".language-level");
  if (checkbox.checked) {
    languageLevelDiv.style.display = "block";
  } else {
    languageLevelDiv.style.display = "none";
  }
}
function showProgrammingLevel(event) {
  var checkbox = event.target;
  // Get the parent element of the checkbox
  var parentDiv = checkbox.parentNode;
  // Find child elements with the programming-level class in the parent element
  var programmingLevelDiv = parentDiv.querySelector(".programming-level");
  if (checkbox.checked) {
    programmingLevelDiv.style.display = "block";
  } else {
    programmingLevelDiv.style.display = "none";
  }
}
function showMajorDirection(selectedMajor) {
  // Hide all MajorDirection containers
  var directionContainers = document.querySelectorAll(".direction-container");
  directionContainers.forEach(function (container) {
    container.style.display = "none";
  });
  var targetDirectionContainer = document.getElementById(
    "direction_" + selectedMajor
  );
  if (targetDirectionContainer) {
    targetDirectionContainer.style.display = "block";
  }
}

window.onload = function () {
  document.querySelectorAll(".toggle-section").forEach((el) => {
    el.style.display = "none";
  });
};

function toggleSection(event, sectionId) {
  var checkbox = event.target;
  var section = document.getElementById(sectionId);
  section.style.display = checkbox.checked ? "block" : "none";
}

document
  .getElementById("viewCourseButton")
  ?.addEventListener("click", function () {
    const courseDetails = localStorage.getItem("radar");
    json = JSON.parse(courseDetails);

    course_id = json.split("'")[3];
    if (courseDetails) {
      // Send the course details to the server
      fetch("/update-course-data", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: courseDetails, // Send course details as JSON
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Course data updated successfully:", data);
          // window.location.href = `/get_single_course/${course_id}/chart`;
        })
        .catch((error) => console.error("Error updating course data:", error));
    } else {
      console.error("No course details found in local storage.");
    }
  });

document.addEventListener("DOMContentLoaded", function () {
  const editBtn = document.getElementById("editbtn");
  const profileContainer = document.getElementById("profileContainer");
  const formContainer = document.getElementById("formContainer");

  if (editBtn) {
    editBtn.addEventListener("click", function (event) {
      event.preventDefault(); // Prevent default link behavior

      // Toggle visibility
      profileContainer.style.display = "none";
      formContainer ? (formContainer.style.display = "block") : "";
    });
  }
});
