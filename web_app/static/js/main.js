document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  const recommendationsSection = document.querySelector(".recommendations");

  if (!form || !recommendationsSection) {
    return;
  }

  form.addEventListener("submit", () => {
    recommendationsSection.classList.add("loading");
    setTimeout(() => recommendationsSection.classList.remove("loading"), 500);
  });
});
