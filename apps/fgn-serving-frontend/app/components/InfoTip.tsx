import { Info } from "lucide-react";

type InfoTipProps = {
  text: string;
};

export function InfoTip({ text }: InfoTipProps) {
  return (
    <span className="info-tip" title={text} aria-label={text}>
      <Info size={12} aria-hidden="true" />
    </span>
  );
}
