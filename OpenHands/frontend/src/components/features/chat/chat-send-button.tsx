import { ArrowUp } from "lucide-react";
import { cn } from "#/utils/utils";

export interface ChatSendButtonProps {
  buttonClassName: string;
  handleSubmit: () => void;
  disabled: boolean;
}

export function ChatSendButton({
  buttonClassName,
  handleSubmit,
  disabled,
}: ChatSendButtonProps) {
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();
    handleSubmit();
  };

  return (
    <button
      type="button"
      className={cn(
        "flex items-center justify-center rounded-full border border-white size-[35px]",
        disabled
          ? "cursor-not-allowed border-neutral-600"
          : "cursor-pointer hover:bg-[#959CB2]",
        buttonClassName,
      )}
      data-name="arrow-up-circle-fill"
      data-testid="submit-button"
      onClick={handleClick}
      disabled={disabled}
    >
      <ArrowUp color={disabled ? "#959CB2" : "white"} />
    </button>
  );
}
