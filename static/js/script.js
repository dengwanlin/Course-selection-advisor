// Ensure DOM is loaded before running the script
document.addEventListener("DOMContentLoaded", () => {
  const menuButton = document.getElementById("menuButton");
  const fullscreenMenu = document.getElementById("fullscreenMenu");
  const closeButton = document.getElementById("closeButton");

  // Open the fullscreen menu
  menuButton.addEventListener("click", () => {
    fullscreenMenu.classList.add("show");
  });

  // Close the fullscreen menu
  closeButton.addEventListener("click", () => {
    fullscreenMenu.classList.remove("show");
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const tabs = {
    searchCoursesTab: document.getElementById("searchCoursesTab"),
    myCoursesTab: document.getElementById("myCoursesTab"),
    allCoursesTab: document.getElementById("allCoursesTab"),
  };

  const contents = {
    searchCoursesForm: document.getElementById("searchCoursesForm"),
    recommendationContent: document.getElementById("recommendationContent"),
    allCoursesContent: document.getElementById("allCoursesContent"),
    myCoursesContent: document.getElementById("myCoursesContent"),
  };

  /**
   * Activate a tab and show corresponding content
   * @param {string} activeTabId - The ID of the tab to activate
   */

  const activateTab = (activeTabId) => {
    // Deactivate all tabs and hide all content
    Object.keys(tabs).forEach((tabId) => {
      tabs[tabId].classList.remove("active");
    });
    Object.keys(contents).forEach((contentId) => {
      contents[contentId].style.display = "none";
    });

    // Activate the selected tab and display its corresponding content
    tabs[activeTabId].classList.add("active");

    if (activeTabId === "searchCoursesTab") {
      if (
        typeof available_recommendations !== "undefined" &&
        available_recommendations == "True"
      ) {
        contents.searchCoursesForm.style.display = "none";
        contents.recommendationContent.style.display = "flex";
      } else {
        contents.searchCoursesForm.style.display = "block";
        contents.recommendationContent.style.display = "none";
      }
    } else if (activeTabId === "myCoursesTab") {
      contents.myCoursesContent.style.display = "flex";
    } else if (activeTabId === "allCoursesTab") {
      contents.allCoursesContent.style.display = "flex";
    }
  };

  // Set the active tab based on the variable passed from Flask
  // Set the active tab based on the variable passed from Flask
  if (typeof activeTab !== "undefined" && tabs[activeTab]) {
    activateTab(activeTab);
  }

  // Add event listeners to tabs
  Object.keys(tabs).forEach((tabId) => {
    tabs[tabId]?.addEventListener("click", () => activateTab(tabId));
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const links = document.querySelectorAll(".sidebar nav a");

  console.log(links);
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
