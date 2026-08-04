/**
 * Render an .excalidraw file to PNG using Excalidraw's own export pipeline, headlessly.
 *
 * Usage: node export_excalidraw.mjs <in.excalidraw> <out.png> [--scale 3] [--padding 10]
 * Normally invoked via regen_api_surface_png.py. See README.md for the node setup.
 *
 * A real browser is required: exportToBlob needs a canvas and Excalidraw's bundled
 * Excalifont/Virgil webfonts. Rendering any other way changes glyph metrics, and the
 * hotspot coords in docs/index.rst are hand-tuned to this rendering.
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const HOME = process.env.EXCALIDRAW_EXPORT_HOME ?? path.join(HERE, "excalidraw-export");
const DIST = path.join(HOME, "node_modules/@excalidraw/excalidraw/dist/prod");
const BUNDLE = path.join(HOME, "bundle.js");

for (const p of [DIST, BUNDLE]) {
  if (!fs.existsSync(p)) {
    console.error(`missing ${p}\nSee docs/tools/README.md for the one-time setup.`);
    process.exit(2);
  }
}
// Resolve puppeteer from HOME via node's normal algorithm, not by guessing its layout.
const puppeteer = createRequire(path.join(HOME, "package.json"))("puppeteer");

const args = process.argv.slice(2);
const [inFile, outFile] = args.filter((a) => !a.startsWith("--"));
const flag = (n, d) => (args.indexOf(`--${n}`) === -1 ? d : Number(args[args.indexOf(`--${n}`) + 1]));
if (!inFile || !outFile) {
  console.error("usage: node export_excalidraw.mjs <in.excalidraw> <out.png> [--scale 3] [--padding 10]");
  process.exit(2);
}
const scale = flag("scale", 3);
const padding = flag("padding", 10);
const scene = JSON.parse(fs.readFileSync(inFile, "utf8"));

const MIME = { ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const url = req.url.split("?")[0];
  if (url === "/") {
    res.writeHead(200, { "Content-Type": "text/html" });
    return res.end(`<!doctype html><meta charset="utf-8"><body>
      <script>window.EXCALIDRAW_ASSET_PATH="/assets/";</script>
      <script src="/bundle.js"></script></body>`);
  }
  if (url === "/bundle.js") {
    res.writeHead(200, { "Content-Type": "text/javascript" });
    return res.end(fs.readFileSync(BUNDLE, "utf8"));
  }
  const f = path.join(DIST, decodeURIComponent(url.replace(/^\/assets\//, "")));
  if (url.startsWith("/assets/") && f.startsWith(DIST) && fs.existsSync(f) && fs.statSync(f).isFile()) {
    res.writeHead(200, { "Content-Type": MIME[path.extname(f)] ?? "application/octet-stream" });
    return res.end(fs.readFileSync(f));
  }
  res.writeHead(404).end("not found");
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));

const browser = await puppeteer.launch({ args: ["--no-sandbox", "--disable-dev-shm-usage"] });
try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("  PAGE ERROR:", e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "load" });
  await page.waitForFunction("window.__excalidrawReady === true", { timeout: 60000 });

  const { b64, fonts } = await page.evaluate(
    async ({ scene, scale, padding }) => {
      const blob = await window.ExcalidrawLib.exportToBlob({
        elements: scene.elements.filter((e) => !e.isDeleted),
        files: scene.files ?? {},
        exportPadding: padding,
        mimeType: "image/png",
        // exportToBlob IGNORES appState.exportScale -- scale must come from
        // getDimensions. Mirrors excalidraw.com: width = (W + 2*padding) * scale.
        getDimensions: (width, height) => ({ width: width * scale, height: height * scale, scale }),
        appState: {
          ...(scene.appState ?? {}),
          exportBackground: true,
          viewBackgroundColor: scene.appState?.viewBackgroundColor ?? "#ffffff",
          exportWithDarkMode: false,
          exportEmbedScene: false,
        },
      });
      await document.fonts.ready;
      const buf = new Uint8Array(await blob.arrayBuffer());
      let bin = "";
      for (let i = 0; i < buf.length; i += 8192) bin += String.fromCharCode(...buf.subarray(i, i + 8192));
      return { b64: btoa(bin), fonts: [...new Set([...document.fonts].filter((f) => f.status === "loaded").map((f) => f.family))] };
    },
    { scene, scale, padding },
  );

  // Webfonts load lazily; a silent fallback substitution shifts every glyph.
  if (!fonts.includes("Excalifont")) {
    console.error(`Excalifont did not load (got: ${fonts.join(", ") || "none"}); refusing to write`);
    process.exit(1);
  }
  fs.writeFileSync(outFile, Buffer.from(b64, "base64"));
  console.log(`wrote ${outFile} (fonts: ${fonts.join(", ")})`);
} finally {
  await browser.close();
  server.close();
}
