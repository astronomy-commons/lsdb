(function () {
  function parseCoords(value) {
    return value
      .split(",")
      .map((part) => Number(part.trim()))
      .filter((part) => Number.isFinite(part));
  }

  function scaleCoords(coords, scale) {
    if (coords.length !== 4) {
      return null;
    }
    return [
      Math.round(coords[0] * scale),
      Math.round(coords[1] * scale),
      Math.round(coords[2] * scale),
      Math.round(coords[3] * scale),
    ];
  }

  function resizeImageMap(image) {
    if (!image.useMap) {
      return;
    }

    const mapName = image.useMap.replace("#", "");
    if (!mapName) {
      return;
    }

    const map = document.querySelector('map[name="' + mapName + '"]');
    if (!map) {
      return;
    }

    const baseWidth = Number(image.dataset.mapWidth || image.naturalWidth);
    const baseHeight = Number(image.dataset.mapHeight || image.naturalHeight);
    if (!Number.isFinite(baseWidth) || !Number.isFinite(baseHeight) || baseWidth <= 0 || baseHeight <= 0) {
      return;
    }

    const renderedWidth = image.clientWidth;
    if (!renderedWidth) {
      return;
    }

    const scale = renderedWidth / baseWidth;

    map.querySelectorAll("area[coords]").forEach((area) => {
      if (!area.dataset.originalCoords) {
        area.dataset.originalCoords = area.getAttribute("coords") || "";
      }
      const original = parseCoords(area.dataset.originalCoords);
      const scaled = scaleCoords(original, scale);
      if (scaled) {
        area.coords = scaled.join(",");
      }
    });
  }

  function initialize() {
    const images = Array.from(document.querySelectorAll("img.api-surface-image[usemap]"));
    if (images.length === 0) {
      return;
    }

    const refresh = () => {
      images.forEach((image) => resizeImageMap(image));
    };

    images.forEach((image) => {
      if (image.complete) {
        resizeImageMap(image);
      } else {
        image.addEventListener("load", () => resizeImageMap(image), { once: true });
      }
    });

    window.addEventListener("resize", refresh);
    window.addEventListener("orientationchange", refresh);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
