import { memo } from "react";

function CornerFrame({
  children,
  className = "",
  markerTopLeft = "X:013 Y:041",
  markerBottomRight = "GRID:A7",
}) {
  return (
    <section className={`corner-frame ${className}`}>
      <div className="corner-layer" aria-hidden="true">
        <span className="corner-bracket corner-bracket-tl" />
        <span className="corner-bracket corner-bracket-tr" />
        <span className="corner-bracket corner-bracket-bl" />
        <span className="corner-bracket corner-bracket-br" />

        <span className="corner-marker corner-marker-tl">{markerTopLeft}</span>
        <span className="corner-marker corner-marker-br">{markerBottomRight}</span>
      </div>

      {children}
    </section>
  );
}

export default memo(CornerFrame);
