#!/usr/bin/env node
"use strict";
/**
 * compute_mofid.cjs -- headless, no-browser MOFid/MOFkey computation.
 *
 * This runs the EXACT same WASM (sbu.wasm, compiled from mofid's own C++
 * SBU-extraction code) and JS (webGavrog, a JS port of Gavrog/Systre used
 * for RCSR topology matching) that power the official web tool at
 * https://snurr-group.github.io/web-mofid/sbu.html -- just driven from
 * Node instead of a browser, with no compiled Zeo++/OpenBabel/Java
 * toolchain to install. The mofid/mofkey string-assembly logic below is
 * ported verbatim from that page's own inline <script> (see
 * `analyzeMOF`/`_extractFragments` in sbu.html), not reimplemented from
 * scratch, so its output should match what the web page itself would
 * show for the same CIF.
 *
 * One real difference from the OFFICIAL PYTHON CLI's output (mofid's own
 * `run_mofid.py:cif2mofid`): that CLI appends a `.{git-commit-ref}`
 * segment to both the MOFid and MOFkey strings for provenance tracking.
 * The web tool's own JS does NOT do this (verified directly against its
 * source), and neither does this script, since it's a faithful port of
 * the web tool specifically. The chemistry (node/linker SMILES fragments,
 * topology symbol) is identical either way -- only that one bookkeeping
 * suffix differs.
 *
 * Usage:
 *   node compute_mofid.cjs path/to/structure.cif
 * Prints one JSON object to stdout on success:
 *   {"mofid": "...", "mofkey": "...", "smiles_nodes": [...],
 *    "smiles_linkers": [...], "topology": "...", "cat": "0"|null,
 *    "cifname": "...", "raw_output": "..."}
 * On failure, prints {"error": "..."} to stdout and exits with code 1 --
 * deliberately still valid JSON on stdout either way, so a Python caller
 * can always `json.loads(stdout)` without checking the exit code first.
 */
global.require = require;
global.module = module;
global.__dirname = __dirname;
global.__filename = __filename;
global.location = { pathname: "/" };

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function fail(message) {
  process.stdout.write(JSON.stringify({ error: message }) + "\n");
  process.exit(1);
}

const cifPath = process.argv[2];
if (!cifPath) {
  fail("usage: node compute_mofid.cjs path/to/structure.cif");
}

let cifData;
try {
  cifData = fs.readFileSync(cifPath, "utf8");
} catch (e) {
  fail(`could not read CIF: ${e.message}`);
}

// ---------------------------------------------------------------------
// Step 1: sbu.wasm -- SBU/fragment extraction (Emscripten module).
// ---------------------------------------------------------------------
const sbuSource = fs.readFileSync(path.join(__dirname, "sbu.js"), "utf8");
const runSbu = new Function(
  "Module",
  sbuSource + "\nModule.FS = FS; Module.HEAPU8 = HEAPU8;\nreturn Module;"
);

let resolveReady;
const ready = new Promise((res) => { resolveReady = res; });
const Module = {
  wasmBinary: fs.readFileSync(path.join(__dirname, "sbu.wasm")),
  noInitialRun: true,
  locateFile: (p) => path.join(__dirname, p),
  getPreloadedPackage: (name) => {
    const buf = fs.readFileSync(name);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  },
  print: () => {},
  printErr: () => {}, // sbu prints a lot of harmless "Cannot write" noise for
                       // folders it doesn't need for this call; silenced here.
  onRuntimeInitialized: () => resolveReady(),
};

let ModuleOut;
try {
  ModuleOut = runSbu(Module);
} catch (e) {
  fail(`failed to initialize sbu.wasm module: ${e.message}`);
}

const TIMEOUT_MS = parseInt(process.env.MOFID_JS_TIMEOUT_MS || "60000", 10);
const watchdog = setTimeout(() => {
  fail(`timed out after ${TIMEOUT_MS}ms waiting for sbu.wasm to initialize`);
}, TIMEOUT_MS);

ready.then(() => {
  clearTimeout(watchdog);
  const FS = ModuleOut.FS;

  try {
    FS.mkdir("/Output");
    FS.mkdir("/Output/MetalOxo");
    FS.mkdir("/Output/SingleNode");
    FS.mkdir("/Output/AllNode");
    FS.mkdir("/Output/StandardIsolated");

    const analyzeMOFc = ModuleOut.cwrap("analyzeMOFc", "null", ["string", "number", "number"]);
    const buflen = 65536;
    const ptr = ModuleOut._malloc(buflen);
    analyzeMOFc(cifData, ptr, buflen);
    const rawOutput = ModuleOut.UTF8ToString(ptr);
    ModuleOut._free(ptr);

    // ---------------------------------------------------------------
    // Step 2: parse fragments -- ported verbatim from sbu.html's
    // `_extractFragments` (see id_constructor.py:extract_fragments for
    // the Python-CLI equivalent this mirrors).
    // ---------------------------------------------------------------
    function extractFragments(rawMofidOutput) {
      let allFragments = rawMofidOutput.trim().split(/\r\n|\r|\n/g).map((a) => a.trim());
      if (allFragments.join("") === "") {
        return { nodes: ["*"], linkers: [], cat: null, parsed: false };
      }
      const catNumLine = allFragments.pop();
      let cat = null;
      if (catNumLine.includes("simplified net(s)")) {
        cat = (parseInt(catNumLine[8], 10) - 1).toString();
        if (cat === "-1") cat = null;
      } else {
        allFragments = ["*"];
      }
      if (allFragments[0] !== "# Nodes:") {
        return { nodes: ["*"], linkers: [], cat, parsed: false };
      }
      allFragments.shift();
      const linkerFlagLoc = allFragments.findIndex((line) => line === "# Linkers:");
      const nodeFragments = allFragments.slice(0, linkerFlagLoc).sort();
      const linkerFragments = allFragments.slice(linkerFlagLoc + 1).sort();
      return { nodes: nodeFragments, linkers: linkerFragments, cat, parsed: true };
    }

    const fragmentInfo = extractFragments(rawOutput);

    // ---------------------------------------------------------------
    // Step 3: webGavrog (JS port of Gavrog/Systre) -- RCSR topology
    // matching against the .cgd nets sbu.wasm just wrote into the
    // virtual filesystem.
    // ---------------------------------------------------------------
    function getTopology(pathToCgd) {
      const rawRcsrExtract = FS.readFile("/RCSRnets.arc", { encoding: "utf8" });
      const rawCgd = FS.readFile(pathToCgd, { encoding: "utf8" });
      if (!rawCgd.includes("NODE")) return "ERROR";
      return webGavrog.getRawRCSRTopology(rawCgd, rawRcsrExtract);
    }

    global.window = global;
    for (const f of ["runtime.js", "vendors.js", "main.js"]) {
      vm.runInThisContext(fs.readFileSync(path.join(__dirname, f), "utf8"), { filename: f });
    }

    const singleNodeTopology = getTopology("/Output/SingleNode/topology.cgd");
    const allNodeTopology = getTopology("/Output/AllNode/topology.cgd");
    let topology;
    if (fragmentInfo.cat === null) {
      topology = "NA";
    } else if (singleNodeTopology === allNodeTopology || allNodeTopology === "ERROR") {
      topology = singleNodeTopology;
    } else {
      topology = singleNodeTopology + "," + allNodeTopology;
    }

    // ---------------------------------------------------------------
    // Step 4: assemble the MOFid/MOFkey strings -- ported verbatim
    // from sbu.html's `analyzeMOF` (see id_constructor.py:assemble_mofid/
    // assemble_mofkey for the Python-CLI equivalents; see this file's
    // header comment for the one deliberate difference, the commit-ref
    // suffix the CLI adds and the web tool does not).
    // ---------------------------------------------------------------
    let mofName = path.basename(cifPath);
    if (mofName.toLowerCase().endsWith(".cif")) {
      mofName = mofName.slice(0, -4);
    }

    const allFragments = fragmentInfo.nodes.concat(fragmentInfo.linkers).sort();
    let mofid = allFragments.join(".");
    mofid += " MOFid-v1.";
    mofid += topology + ".";
    if (fragmentInfo.cat === "no_mof") {
      mofid += fragmentInfo.cat;
    } else if (fragmentInfo.cat !== null) {
      mofid += "cat" + fragmentInfo.cat;
    } else {
      mofid += "NA";
    }
    if (mofid.startsWith(" ")) {
      mofid = "*" + mofid + "no_mof";
    }
    mofid += ";" + mofName;

    let baseMofkey = "";
    if (fragmentInfo.parsed) {
      try {
        baseMofkey = FS.readFile("/Output/MetalOxo/mofkey_no_topology.txt", { encoding: "utf8" }).trim();
      } catch (e) {
        baseMofkey = "";
      }
    }
    const mofkey = baseMofkey.replace("MOFkey-v1", "MOFkey-v1." + singleNodeTopology);

    process.stdout.write(
      JSON.stringify({
        mofid,
        mofkey,
        smiles_nodes: fragmentInfo.nodes,
        smiles_linkers: fragmentInfo.linkers,
        topology,
        cat: fragmentInfo.cat,
        cifname: mofName,
        raw_output: rawOutput,
        parsed: fragmentInfo.parsed,
      }) + "\n"
    );
    process.exit(fragmentInfo.parsed ? 0 : 1);
  } catch (e) {
    fail(`${e.name || "Error"}: ${e.message}`);
  }
}).catch((e) => fail(`sbu.wasm never became ready: ${e.message}`));
