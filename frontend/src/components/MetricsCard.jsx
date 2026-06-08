import {
    Card,
    Typography
} from "@mui/material";

export default function MetricsCard() {

    return (
        <Card sx={{ p: 3 }}>

            <Typography variant="h6">
                Model Metrics
            </Typography>

            <Typography mt={2}>
                mIoU: 0.5143
            </Typography>

            <Typography>
                Model: SegFormer-B0
            </Typography>

            <Typography>
                Classes: 5
            </Typography>

        </Card>
    );
}