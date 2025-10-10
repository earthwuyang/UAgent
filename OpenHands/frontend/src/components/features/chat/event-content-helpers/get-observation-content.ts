import {
  ReadObservation,
  CommandObservation,
  IPythonObservation,
  EditObservation,
  BrowseObservation,
  OpenHandsObservation,
  RecallObservation,
  TaskTrackingObservation,
} from "#/types/core/observations";
import { getObservationResult } from "./get-observation-result";
import { getDefaultEventContent, MAX_CONTENT_LENGTH } from "./shared";
import i18n from "#/i18n";

const getReadObservationContent = (event: ReadObservation): string =>
  `\`\`\`\n${event.content}\n\`\`\``;

const getCommandObservationContent = (
  event: CommandObservation | IPythonObservation,
): string => {
  let { content } = event;
  if (content.length > MAX_CONTENT_LENGTH) {
    content = `${content.slice(0, MAX_CONTENT_LENGTH)}...`;
  }
  return `Output:\n\`\`\`sh\n${content.trim() || i18n.t("OBSERVATION$COMMAND_NO_OUTPUT")}\n\`\`\``;
};

const getEditObservationContent = (
  event: EditObservation,
  successMessage: boolean,
): string => {
  if (successMessage) {
    return `\`\`\`diff\n${event.extras.diff}\n\`\`\``; // Content is already truncated by the ACI
  }
  return event.content;
};

const getBrowseObservationContent = (event: BrowseObservation) => {
  let contentDetails = `**URL:** ${event.extras.url}\n`;
  if (event.extras.error) {
    contentDetails += `\n\n**Error:**\n${event.extras.error}\n`;
  }
  contentDetails += `\n\n**Output:**\n${event.content}`;
  if (contentDetails.length > MAX_CONTENT_LENGTH) {
    contentDetails = `${contentDetails.slice(0, MAX_CONTENT_LENGTH)}...(truncated)`;
  }
  return contentDetails;
};

const getRecallObservationContent = (event: RecallObservation): string => {
  let content = "";

  if (event.extras.recall_type === "workspace_context") {
    if (event.extras.repo_name) {
      content += `\n\n**Repository:** ${event.extras.repo_name}`;
    }
    if (event.extras.repo_directory) {
      content += `\n\n**Directory:** ${event.extras.repo_directory}`;
    }
    if (event.extras.date) {
      content += `\n\n**Date:** ${event.extras.date}`;
    }
    if (
      event.extras.runtime_hosts &&
      Object.keys(event.extras.runtime_hosts).length > 0
    ) {
      content += `\n\n**Available Hosts**`;
      for (const [host, port] of Object.entries(event.extras.runtime_hosts)) {
        content += `\n\n- ${host} (port ${port})`;
      }
    }
    if (event.extras.repo_instructions) {
      content += `\n\n**Repository Instructions:**\n\n${event.extras.repo_instructions}`;
    }
    if (event.extras.conversation_instructions) {
      content += `\n\n**Conversation Instructions:**\n\n${event.extras.conversation_instructions}`;
    }
    if (event.extras.additional_agent_instructions) {
      content += `\n\n**Additional Instructions:**\n\n${event.extras.additional_agent_instructions}`;
    }
  }

  // Handle microagent knowledge
  if (
    event.extras.microagent_knowledge &&
    event.extras.microagent_knowledge.length > 0
  ) {
    content += `\n\n**Triggered Microagent Knowledge:**`;
    for (const knowledge of event.extras.microagent_knowledge) {
      content += `\n\n- **${knowledge.name}** (triggered by keyword: ${knowledge.trigger})\n\n${knowledge.content}`;
    }
  }

  if (
    event.extras.custom_secrets_descriptions &&
    Object.keys(event.extras.custom_secrets_descriptions).length > 0
  ) {
    content += `\n\n**Custom Secrets**`;
    for (const [name, description] of Object.entries(
      event.extras.custom_secrets_descriptions,
    )) {
      content += `\n\n- $${name}: ${description}`;
    }
  }

  return content;
};

const getTaskTrackingObservationContent = (
  event: TaskTrackingObservation,
): string => {
  const { command, task_list: taskList } = event.extras;
  let content = `**Command:** \`${command}\``;

  if (command === "plan" && taskList.length > 0) {
    content += `\n\n**Task List (${taskList.length} ${taskList.length === 1 ? "item" : "items"}):**\n`;

    taskList.forEach((task, index) => {
      const statusIcon =
        {
          todo: "⏳",
          in_progress: "🔄",
          done: "✅",
        }[task.status] || "❓";

      content += `\n${index + 1}. ${statusIcon} **[${task.status.toUpperCase().replace("_", " ")}]** ${task.title}`;
      content += `\n   *ID: ${task.id}*`;
      if (task.notes) {
        content += `\n   *Notes: ${task.notes}*`;
      }
    });
  } else if (command === "plan") {
    content += "\n\n**Task List:** Empty";
  }

  if (event.content && event.content.trim()) {
    content += `\n\n**Result:** ${event.content.trim()}`;
  }

  return content;
};

const getSubAgentSpawnedContent = (event: SubAgentSpawnedObservation): string => {
  const { sub_agent_id, sub_agent_type, goal } = event.extras;
  const truncatedGoal = goal.length > 200 ? goal.substring(0, 200) + "..." : goal;
  return `Sub-agent ${sub_agent_id} (${sub_agent_type}) spawned with goal: ${truncatedGoal}`;
};

const getSubAgentProgressContent = (event: SubAgentProgressObservation): string => {
  const { sub_agent_id, status, progress, current_task, tree_stats } = event.extras;
  const progressPercent = Math.round(progress * 100);
  let content = `Status: ${status}, Progress: ${progressPercent}%, Task: ${current_task}`;
  
  if (tree_stats) {
    const stats = [];
    if (tree_stats.total_nodes !== undefined) stats.push(`Nodes: ${tree_stats.total_nodes}`);
    if (tree_stats.total_cost !== undefined) stats.push(`Cost: $${tree_stats.total_cost.toFixed(2)}`);
    if (stats.length > 0) {
      content += `\n${stats.join(", ")}`;
    }
  }
  
  return content;
};

const getSubAgentCompletedContent = (event: SubAgentCompletedObservation): string => {
  const { sub_agent_id, status, result, tree_stats } = event.extras;
  let content = `Sub-agent ${sub_agent_id} ${status}`;
  
  if (result) {
    content += `: ${result}`;
  }
  
  if (tree_stats) {
    const stats = [];
    if (tree_stats.completed_nodes !== undefined) stats.push(`${tree_stats.completed_nodes} nodes`);
    if (tree_stats.iterations !== undefined) stats.push(`${tree_stats.iterations} iterations`);
    if (stats.length > 0) {
      content += `\nCompleted ${stats.join(" in ")}`;
    }
  }
  
  return content;
};

export const getObservationContent = (event: OpenHandsObservation): string => {
  switch (event.observation) {
    case "read":
      return getReadObservationContent(event);
    case "edit":
      return getEditObservationContent(
        event,
        getObservationResult(event) === "success",
      );
    case "run_ipython":
    case "run":
      return getCommandObservationContent(event);
    case "browse":
      return getBrowseObservationContent(event);
    case "recall":
      return getRecallObservationContent(event);
    case "task_tracking":
      return getTaskTrackingObservationContent(event);
    case "sub_agent_spawned":
      return getSubAgentSpawnedContent(event as SubAgentSpawnedObservation);
    case "sub_agent_progress":
      return getSubAgentProgressContent(event as SubAgentProgressObservation);
    case "sub_agent_completed":
      return getSubAgentCompletedContent(event as SubAgentCompletedObservation);
    default:
      return getDefaultEventContent(event);
  }
};
