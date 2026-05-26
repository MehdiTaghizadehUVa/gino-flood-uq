import { InfoTip } from "./InfoTip";

type InsightCaptionProps = {
  caption: string;
  insight?: string;
};

export function InsightCaption({ caption, insight }: InsightCaptionProps) {
  return (
    <figcaption className="figure-caption">
      <strong>{caption}</strong>
      {insight ? (
        <>
          {" "}
          <span>{insight}</span>
          <InfoTip text={insight} />
        </>
      ) : null}
    </figcaption>
  );
}
