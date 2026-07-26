// Shared types for Tasknote's frontend - mirrors the shapes returned by
// the HTTP API in api.js so the frontend gets type-checked access without
// duplicating the definitions by hand.

export type Priority = "low" | "medium" | "high";

export interface Tag {
  name: string;
  color: string;
}

export interface Task {
  id: number;
  title: string;
  tag: string | null;
  priority: Priority;
  done: boolean;
  dueDate: string | null;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
}

export function isOverdue(task: Task, today: Date): boolean {
  if (task.done || task.dueDate === null) {
    return false;
  }
  return new Date(task.dueDate) < today;
}
