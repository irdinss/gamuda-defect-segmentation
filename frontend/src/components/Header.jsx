import {
    Card,
    Typography,
    Chip,
    Stack
} from "@mui/material";

export default function Header() {

    return (
        <Card
            sx={{
                p: 3,
                mb: 3,
            }}
        >
            <Stack
                direction="row"
                justifyContent="space-between"
            >
                <Typography variant="h4">
                    Infrastructure Defect Segmentation
                </Typography>

                <Chip
                    label="Online"
                    color="success"
                />
            </Stack>
        </Card>
    );
}