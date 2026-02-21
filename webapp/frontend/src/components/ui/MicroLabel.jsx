import { memo } from "react";

function MicroLabel({ children, className = "" }) {
  return <p className={`micro-label ${className}`}>[ {children} ]</p>;
}

export default memo(MicroLabel);
