import {
    Card,
    Typography,
    LinearProgress,
    Stack
} from "@mui/material";

export default function StatisticsCard({
    stats
}) {

    return (
        <Card sx={{ p: 3 }}>

            <Typography variant="h6">
                Defect Statistics
            </Typography>

            {Object.entries(stats).map(
                ([key, value]) => (

                    <Stack
                        key={key}
                        sx={{ mt: 2 }}
                    >

                        <Typography>
                            {key}: {value}%
                        </Typography>

                        <LinearProgress
                            variant="determinate"
                            value={value}
                        />

                    </Stack>
                )
            )}

        </Card>
    );
}