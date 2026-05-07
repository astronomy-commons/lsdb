(function () {
  function parseCoords(value) {
    return String(value || "")
      .split(",")
      .map((part) => Number(part.trim()))
      .filter((part) => Number.isFinite(part));
  }

  function scaleRect(coords, scaleX, scaleY) {
    if (coords.length !== 4) {
      return null;
    }
    return [
      Math.round(coords[0] * scaleX),
      Math.round(coords[1] * scaleY),
      Math.round(coords[2] * scaleX),
      Math.round(coords[3] * scaleY),
    ];
  }

  function getMapByImage(image) {
    if (!image || !image.useMap) {
      return null;
    }
    const mapName = image.useMap.replace("#", "");
    if (!mapName) {
      return null;
    }

    return document.querySelector('map[name="' + mapName + '"]');
  }

  function ensureAuthoredCoords(area) {
    if (!area.dataset.authoredCoords) {
      area.dataset.authoredCoords = area.getAttribute("coords") || "";
    }

    return parseCoords(area.dataset.authoredCoords);
  }

  function getDimensions(image) {
    const baseWidth = Number(image.dataset.mapWidth || image.naturalWidth);
    const baseHeight = Number(image.dataset.mapHeight || image.naturalHeight);
    const authoredWidth = Number(image.dataset.originalMapWidth || baseWidth);
    const authoredHeight = Number(image.dataset.originalMapHeight || baseHeight);

    if (
      !Number.isFinite(baseWidth) ||
      !Number.isFinite(baseHeight) ||
      !Number.isFinite(authoredWidth) ||
      !Number.isFinite(authoredHeight) ||
      baseWidth <= 0 ||
      baseHeight <= 0 ||
      authoredWidth <= 0 ||
      authoredHeight <= 0
    ) {
      return null;
    }

    return {
      baseWidth,
      baseHeight,
      authoredWidth,
      authoredHeight,
    };
  }

  function getRenderedSize(image, dims) {
    const renderedWidth = image.clientWidth;
    if (!renderedWidth || renderedWidth <= 0) {
      return null;
    }

    const renderedHeight =
      image.clientHeight ||
      image.getBoundingClientRect().height ||
      (renderedWidth * dims.baseHeight) / dims.baseWidth;
    if (!renderedHeight || renderedHeight <= 0) {
      return null;
    }

    return {
      renderedWidth,
      renderedHeight,
    };
  }

  function getRenderedRect(area, dims, rendered) {
    const authored = ensureAuthoredCoords(area);
    if (authored.length !== 4) {
      return null;
    }

    const toBaseX = dims.baseWidth / dims.authoredWidth;
    const toBaseY = dims.baseHeight / dims.authoredHeight;
    const baseRect = scaleRect(authored, toBaseX, toBaseY);
    if (!baseRect) {
      return null;
    }

    const toRenderedX = rendered.renderedWidth / dims.baseWidth;
    const toRenderedY = rendered.renderedHeight / dims.baseHeight;
    const renderedRect = scaleRect(baseRect, toRenderedX, toRenderedY);
    return renderedRect ? shrinkRect(renderedRect, rendered.image) : null;
  }

  function shrinkRect(rect, image) {
    const scaleX = Number(image.dataset.hitboxScaleX || 1);
    const scaleY = Number(image.dataset.hitboxScaleY || 1);
    const sx = Number.isFinite(scaleX) && scaleX > 0 && scaleX <= 1 ? scaleX : 1;
    const sy = Number.isFinite(scaleY) && scaleY > 0 && scaleY <= 1 ? scaleY : 1;

    const x1 = Math.min(rect[0], rect[2]);
    const y1 = Math.min(rect[1], rect[3]);
    const x2 = Math.max(rect[0], rect[2]);
    const y2 = Math.max(rect[1], rect[3]);

    const width = x2 - x1;
    const height = y2 - y1;
    const cx = x1 + width / 2;
    const cy = y1 + height / 2;

    const minWidth = 24;
    const minHeight = 8;
    const newWidth = Math.max(minWidth, width * sx);
    const newHeight = Math.max(minHeight, height * sy);

    return [
      Math.round(cx - newWidth / 2),
      Math.round(cy - newHeight / 2),
      Math.round(cx + newWidth / 2),
      Math.round(cy + newHeight / 2),
    ];
  }

  function clearDebugLayer(wrapper) {
    const existing = wrapper.querySelector(".api-surface-debug-layer");
    if (existing) {
      existing.remove();
    }
  }

  function isDebugEnabled() {
    const params = new URLSearchParams(window.location.search);
    return params.get("mapdebug") === "1";
  }

  function renderDebugOverlay(image, map, dims, rendered) {
    const wrapper = image.closest(".api-surface-wrapper");
    if (!wrapper) {
      return;
    }

    clearDebugLayer(wrapper);
    if (!isDebugEnabled()) {
      return;
    }

    const layer = document.createElement("div");
    layer.className = "api-surface-debug-layer";
    wrapper.appendChild(layer);

    map.querySelectorAll("area[coords]").forEach((area) => {
      const rect = getRenderedRect(area, dims, rendered);
      if (!rect) {
        return;
      }

      const left = Math.min(rect[0], rect[2]);
      const top = Math.min(rect[1], rect[3]);
      const width = Math.abs(rect[2] - rect[0]);
      const height = Math.abs(rect[3] - rect[1]);

      const box = document.createElement("div");
      box.className = "api-surface-debug-box";
      box.style.left = left + "px";
      box.style.top = top + "px";
      box.style.width = width + "px";
      box.style.height = height + "px";

      layer.appendChild(box);
    });
  }

  function canUseMagnifier() {
    if (!window.matchMedia) {
      return true;
    }
    const hasFinePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    return hasFinePointer;
  }

  function getMagnifierConfig(image) {
    const zoom = Number(image.dataset.magnifyZoom || 2);
    const size = Number(image.dataset.magnifySize || 220);
    return {
      zoom: Number.isFinite(zoom) && zoom > 1 ? zoom : 2,
      size: Number.isFinite(size) && size >= 100 ? size : 220,
    };
  }

  function ensureMagnifier(image) {
    if (image._apiSurfaceMagnifier) {
      return image._apiSurfaceMagnifier;
    }
    const wrapper = image.closest(".api-surface-wrapper");
    if (!wrapper) {
      return null;
    }
    const lens = wrapper.querySelector(".api-surface-magnifier");
    if (!lens) {
      return null;
    }

    const config = getMagnifierConfig(image);
    const state = {
      wrapper,
      lens,
      zoom: config.zoom,
      size: config.size,
      active: false,
      lastClientX: null,
      lastClientY: null,
    };
    state.lens.style.width = `${state.size}px`;
    state.lens.style.height = `${state.size}px`;
    image._apiSurfaceMagnifier = state;
    return state;
  }

  function setMagnifierBackground(image, state) {
    const src = image.currentSrc || image.getAttribute("src");
    if (!src) {
      return;
    }
    state.lens.style.backgroundImage = `url("${src}")`;
  }

  function updateMagnifier(image, clientX, clientY) {
    const state = ensureMagnifier(image);
    if (!state || !state.active) {
      return;
    }

    const imageRect = image.getBoundingClientRect();
    if (!imageRect.width || !imageRect.height) {
      return;
    }
    const wrapperRect = state.wrapper.getBoundingClientRect();
    const radius = state.size / 2;

    const localX = Math.max(0, Math.min(imageRect.width, clientX - imageRect.left));
    const localY = Math.max(0, Math.min(imageRect.height, clientY - imageRect.top));

    const imageLeft = imageRect.left - wrapperRect.left;
    const imageTop = imageRect.top - wrapperRect.top;
    const minLeft = imageLeft;
    const maxLeft = imageLeft + imageRect.width - state.size;
    const minTop = imageTop;
    const maxTop = imageTop + imageRect.height - state.size;

    const desiredLeft = imageLeft + localX - radius;
    const desiredTop = imageTop + localY - radius;
    const lensLeft = Math.max(minLeft, Math.min(maxLeft, desiredLeft));
    const lensTop = Math.max(minTop, Math.min(maxTop, desiredTop));

    state.lens.style.left = `${lensLeft}px`;
    state.lens.style.top = `${lensTop}px`;
    state.lens.style.backgroundSize = `${imageRect.width * state.zoom}px ${imageRect.height * state.zoom}px`;

    const bgX = -(localX * state.zoom - radius);
    const bgY = -(localY * state.zoom - radius);
    state.lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
  }

  function attachMagnifier(image) {
    if (image.dataset.magnifierReady === "1") {
      return;
    }
    if (!canUseMagnifier()) {
      return;
    }

    const state = ensureMagnifier(image);
    if (!state) {
      return;
    }
    setMagnifierBackground(image, state);

    const hide = () => {
      state.active = false;
      state.wrapper.classList.remove("is-magnifying");
    };

    const show = (event) => {
      state.active = true;
      state.wrapper.classList.add("is-magnifying");
      if (event) {
        state.lastClientX = event.clientX;
        state.lastClientY = event.clientY;
        updateMagnifier(image, event.clientX, event.clientY);
      }
    };

    image.addEventListener("pointerenter", show);
    image.addEventListener("pointermove", (event) => {
      if (!state.active) {
        return;
      }
      state.lastClientX = event.clientX;
      state.lastClientY = event.clientY;
      updateMagnifier(image, event.clientX, event.clientY);
    });
    image.addEventListener("pointerleave", hide);
    image.addEventListener("pointercancel", hide);

    image.dataset.magnifierReady = "1";
  }

  function refreshMagnifier(image) {
    const state = ensureMagnifier(image);
    if (!state) {
      return;
    }
    setMagnifierBackground(image, state);
    if (state.active && state.lastClientX !== null && state.lastClientY !== null) {
      updateMagnifier(image, state.lastClientX, state.lastClientY);
    }
  }

  function resizeImageMap(image) {
    const map = getMapByImage(image);
    if (!map) {
      return;
    }

    const dims = getDimensions(image);
    if (!dims) {
      return;
    }

    const rendered = getRenderedSize(image, dims);
    if (!rendered) {
      return;
    }
    rendered.image = image;

    map.querySelectorAll("area[coords]").forEach((area) => {
      const rect = getRenderedRect(area, dims, rendered);
      if (rect) {
        area.coords = rect.join(",");
      }
    });

    renderDebugOverlay(image, map, dims, rendered);
    refreshMagnifier(image);
  }

  function prepareImageForInteraction(image) {
    if (image.dataset.interactionReady === "1") {
      return;
    }

    image.dataset.interactionReady = "1";
    image.setAttribute("draggable", "false");
    image.addEventListener("dragstart", (event) => {
      event.preventDefault();
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
      prepareImageForInteraction(image);
      attachMagnifier(image);
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
