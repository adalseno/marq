// Tasknote's HTTP API - a thin wrapper that forwards each request to the
// Python core over a local socket and serializes the response as JSON.
// No business logic lives here on purpose.

const http = require("http");

function handleListTasks(req, res, core) {
  const url = new URL(req.url, "http://localhost");
  const tag = url.searchParams.get("tag");
  core.request("list_tasks", { tag }).then((tasks) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(tasks));
  });
}

function handleAddTask(req, res, core) {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk;
  });
  req.on("end", () => {
    const { title, tag, priority } = JSON.parse(body);
    core.request("add_task", { title, tag, priority }).then((taskId) => {
      res.writeHead(201, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ id: taskId }));
    });
  });
}

function createServer(core) {
  return http.createServer((req, res) => {
    if (req.method === "GET" && req.url.startsWith("/tasks")) {
      return handleListTasks(req, res, core);
    }
    if (req.method === "POST" && req.url === "/tasks") {
      return handleAddTask(req, res, core);
    }
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
  });
}

module.exports = { createServer };
