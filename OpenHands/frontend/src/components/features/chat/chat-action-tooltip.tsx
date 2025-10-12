import { Tooltip } from "@heroui/react";

interface ChatActionTooltipProps {
  children: React.ReactNode;
  tooltip: string | React.ReactNode;
  ariaLabel: string;
}

export function ChatActionTooltip({
  children,
  tooltip,
  ariaLabel,
}: ChatActionTooltipProps) {
  return (
    <Tooltip
      content={tooltip}
      closeDelay={100}
      placement="bottom"
      className="bg-white text-black text-xs font-medium leading-5"
      aria-label={ariaLabel}
    >
      {children}
    </Tooltip>
  );
}
