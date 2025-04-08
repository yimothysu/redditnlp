import PopupState, { bindTrigger, bindPopover } from "material-ui-popup-state";
import Button from "@mui/material/Button";
import { Typography } from "@mui/material";
import Popover from "@mui/material/Popover";

export default function ButtonPopover({ title, children }) {
            return (
                <PopupState variant="popover" popupId="demo-popup-popover">
                    {(popupState) => (
                        <div>
                            <Button
                                sx={{
                                    backgroundColor: "#4f46e5",
                                    mb: 0,
                                    fontSize: "14px",
                                    transition:
                                        "background-color 0.2s ease-in-out",
                                    "&:hover": { backgroundColor: "#4338ca" },
                                }}
                                variant="contained"
                                {...bindTrigger(popupState)}
                            >
                                { title }
                            </Button>
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
                                    { children }
                                </Typography>
                            </Popover>
                        </div>
                    )}
                </PopupState>
            );
        };