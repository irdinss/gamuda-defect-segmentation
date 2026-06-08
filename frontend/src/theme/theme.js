import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
    palette: {
        mode: "dark",

        background: {
            default: "#020617",
            paper: "#0f172a",
        },

        primary: {
            main: "#38bdf8",
        },
    },

    shape: {
        borderRadius: 16,
    },
});