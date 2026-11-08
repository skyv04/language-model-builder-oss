// Client for the local engine API served by model_builder/server.py.

async function request(method, path, body) {
  const options = { method, credentials: "same-origin", headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: response.ok ? text : `${response.status} ${response.statusText || "request failed"}` };
  }
  if (!response.ok) {
    const error = new Error((data && data.error) || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

async function stream(path, body, onEvent, signal) {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
    signal,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      message = (await response.json()).error || message;
    } catch {
      /* keep status text */
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

async function upload(path, file, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(`${path}${query ? `?${query}` : ""}`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) },
    body: file,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

export const api = {
  get: (path) => request("GET", path),
  post: (path, body = {}) => request("POST", path, body),
  patch: (path, body = {}) => request("PATCH", path, body),
  del: (path) => request("DELETE", path),
  stream,
  upload,
};

const listeners = new Map();
let source = null;
let retryDelay = 1000;

function dispatch(message) {
  for (const key of [message.type, "*"]) {
    for (const fn of listeners.get(key) || []) {
      try {
        fn(message);
      } catch (error) {
        console.error(error);
      }
    }
  }
}

function connect() {
  source = new EventSource("/api/events");
  source.onopen = () => {
    retryDelay = 1000;
    dispatch({ type: "connection", connected: true });
  };
  source.onmessage = (event) => dispatch(JSON.parse(event.data));
  source.onerror = () => {
    source.close();
    dispatch({ type: "connection", connected: false });
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 10000);
  };
}

export const events = {
  on(type, fn) {
    if (!listeners.has(type)) listeners.set(type, new Set());
    listeners.get(type).add(fn);
    return () => listeners.get(type).delete(fn);
  },
  start() {
    if (!source) connect();
  },
  emit: dispatch,
};
