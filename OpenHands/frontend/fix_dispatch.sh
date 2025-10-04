#!/bin/bash

# Fix dispatch wrappers around Zustand functions
# Pattern: dispatch(functionName(...)) -> functionName(...)

files=(
"src/components/features/browser/browser.tsx"
"src/components/features/chat/chat-interface.tsx"
"src/components/features/chat/custom-chat-input.tsx"
"src/components/features/chat/interactive-chat-box.tsx"
"src/components/features/chat/uploaded-files.tsx"
"src/components/features/controls/agent-status.tsx"
"src/components/features/controls/git-tools-submenu.tsx"
"src/components/features/controls/macros-submenu.tsx"
"src/components/features/conversation/conversation-tabs/conversation-tabs.tsx"
"src/components/features/microagent-management/microagent-management-add-microagent-button.tsx"
"src/components/features/microagent-management/microagent-management-content.tsx"
"src/components/features/microagent-management/microagent-management-learn-this-repo.tsx"
"src/components/features/microagent-management/microagent-management-microagent-card.tsx"
"src/components/features/microagent-management/microagent-management-sidebar.tsx"
"src/components/features/microagent-management/microagent-management-view-microagent-header.tsx"
"src/components/shared/buttons/confirmation-buttons.tsx"
"src/hooks/chat/use-chat-input-logic.ts"
"src/hooks/chat/use-grip-resize.ts"
)

for file in "${files[@]}"; do
  echo "Processing $file..."

  # Remove dispatch() wrappers - match dispatch(functionName(...))
  # This handles function calls with any arguments
  sed -i 's/dispatch(\(clear[A-Za-z]*([^)]*)\))/\1/g' "$file"
  sed -i 's/dispatch(\(set[A-Za-z]*([^)]*)\))/\1/g' "$file"
  sed -i 's/dispatch(\(add[A-Za-z]*([^)]*)\))/\1/g' "$file"
  sed -i 's/dispatch(\(remove[A-Za-z]*([^)]*)\))/\1/g' "$file"
  sed -i 's/dispatch(\(reset[A-Za-z]*([^)]*)\))/\1/g' "$file"
done

echo "Done!"
