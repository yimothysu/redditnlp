import PopupState, { bindTrigger, bindPopover } from "material-ui-popup-state";
import Button from "@mui/material/Button";
import { Typography } from "@mui/material";
import Popover from "@mui/material/Popover";
import React from "react";

interface Props {
  title: string;
  children: React.ReactNode;
}

// This component creates a button with a specified title and places the content between the ButtonPopover tags in the content
// of the popover that displays when the button is pressed
export default function ButtonPopover({ title, children }: Props) {
  return (
    <PopupState variant="popover" popupId="demo-popup-popover">
      {(popupState) => (
        <div>
          <button
            className="bg-[#fa6f4d] font-medium text-sm transition-colors duration-200 hover:bg-[#e36748] px-4 py-2 rounded-md text-white hover:cursor-pointer"
            onClick={() => popupState.open()}
          >
            {title}
          </button>
          <Popover
            {...bindPopover(popupState)}
            anchorOrigin={{
              vertical: "bottom",
              horizontal: "center",
            }}
            transformOrigin={{
              vertical: "top",
              horizontal: "center",
            }}
          >
            <Typography
              sx={{
                p: 2,
                maxWidth: 500,
                fontSize: "14px",
              }}
            >
              {children}
            </Typography>
          </Popover>
        </div>
      )}
    </PopupState>
  );
}
